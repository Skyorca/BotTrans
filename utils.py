import sys
sys.path.append('../..')
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, \
    roc_auc_score, precision_recall_curve, auc, average_precision_score
import torch
import torch.nn.functional as func
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.var = 0
        self.all_val = []
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.all_val.append(val)
        all_val = np.array(self.all_val)
        self.var = all_val.std()
        self.max = max(self.all_val)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def null_metrics():
    return {
        'acc': 0.0,
        'f1-score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'mcc': 0.0,
        'roc-auc': 0.0,
        'pr-auc': 0.0
    }


def prTopK(y, pred_score, k=0.1):
    topk = int(k * len(y))
    All_P = torch.sum(y)
    _, pred_topk_idx = torch.topk(pred_score, k=topk)
    gt_idx = torch.nonzero(y).squeeze()
    pred = set(pred_topk_idx.numpy())
    gt = set(gt_idx.numpy())
    TP = len(pred.intersection(gt))
    p_k = TP / topk
    r_k = TP / All_P
    f_k = 2 * p_k * r_k / (p_k + r_k)
    # gt_topk_label = y[pred_topk_idx]
    # TP = torch.sum(gt_topk_label)
    return p_k, r_k, f_k


def calc_metrics(y, pred, k=0.1):
    """
    :param y: groudtruth (1,L)
    :param pred: pred probs (L,2) not normalized to probs
    :return:
    """
    assert y.dim() == 1 and pred.dim() == 2
    if torch.any(torch.isnan(pred)):
        metrics = null_metrics()
        plog = ''
        for key, value in metrics.items():
            plog += ' {}: {:.6}'.format(key, value)
        return metrics, plog
    pred = func.softmax(pred, dim=-1)
    pred_label = torch.argmax(pred, dim=-1)
    pred_score = pred[:, -1]
    y = y.to('cpu').numpy().tolist()
    pred_label = pred_label.to('cpu').tolist()
    pred_score = pred_score.to('cpu').tolist()
    precision, recall, _thresholds = precision_recall_curve(y, pred_score)
    preTopK, recallTopK, f1TopK = prTopK(y=torch.tensor(y), pred_score=torch.tensor(pred_score), k=k)
    metrics = {
        'acc': accuracy_score(y, pred_label),
        'f1-score': f1_score(y, pred_label),
        'precision': precision_score(y, pred_label),
        'recall': recall_score(y, pred_label),
        'mcc': matthews_corrcoef(y, pred_label),
        'roc-auc': roc_auc_score(y, pred_score),
        'pr-auc': auc(recall, precision),
        'ap': average_precision_score(y_true=y, y_score=pred_score),
        'precision@K': preTopK,
        'recall@K': recallTopK,
        'f1@K': f1TopK
    }
    plog = ''
    for key in ['f1-score', 'roc-auc', 'ap']:
        plog += ' {}: {:.6}'.format(key, metrics[key])
    return metrics, plog


def calc_metrics_for_anomalvals(y, pred_score, k=0.1):
    """
    :param y: groudtruth (1,L)
    :param pred: anomaly value in [0,1]: (L,1)
    :return:
    """
    assert y.dim() == 1 and pred_score.dim() == 1
    if torch.any(torch.isnan(pred_score)):
        metrics = null_metrics()
        plog = ''
        for key, value in metrics.items():
            plog += ' {}: {:.6}'.format(key, value)
        return metrics, plog

    threshold = 0.5
    pred_label = (pred_score > threshold).long()
    y = y.to('cpu').numpy().tolist()
    pred_label = pred_label.to('cpu').tolist()
    pred_score = pred_score.to('cpu').tolist()
    precision, recall, _thresholds = precision_recall_curve(y, pred_score)
    preTopK, recallTopK, f1TopK = prTopK(y=torch.tensor(y), pred_score=torch.tensor(pred_score), k=k)
    metrics = {
        'acc': accuracy_score(y, pred_label),
        'f1-score': f1_score(y, pred_label),
        'precision': precision_score(y, pred_label),
        'recall': recall_score(y, pred_label),
        'mcc': matthews_corrcoef(y, pred_label),
        'roc-auc': roc_auc_score(y, pred_score),
        'pr-auc': auc(recall, precision),
        'ap': average_precision_score(y_true=y, y_score=pred_score),
        'precision@K': preTopK,
        'recall@K': recallTopK,
        'f1@K': f1TopK
    }
    plog = ''
    for key in ['f1-score', 'roc-auc', 'ap']:
        plog += ' {}: {:.6}'.format(key, metrics[key])
    return metrics, plog


def is_better(now, pre):
    if now['f1-score'] != pre['f1-score']:
        return now['f1-score'] > pre['f1-score']
    if now['roc-auc'] != pre['roc-auc']:
        return now['roc-auc'] > pre['roc-auc']
    if now['precision@K'] != pre['precision@K']:
        return now['precision@K'] > pre['precision@K']
    if now['recall@K'] != pre['recall@K']:
        return now['recall@K'] > pre['recall@K']
    if now['acc'] != pre['acc']:
        return now['acc'] > pre['acc']
    if now['mcc'] != pre['mcc']:
        return now['mcc'] > pre['mcc']
    if now['pr-auc'] != pre['pr-auc']:
        return now['pr-auc'] > pre['pr-auc']
    if now['precision'] != pre['precision']:
        return now['precision'] > pre['precision']
    if now['recall'] != pre['recall']:
        return now['recall'] > pre['recall']
    return False


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class GaussianKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)


        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix

def compute_dist(src_z:torch.Tensor, tgt_z:torch.Tensor):
    if src_z.shape[0] != tgt_z.shape[0]:
        size = min(src_z.shape[0],tgt_z.shape[0])
        random_indices = np.random.permutation(size)
        x = src_z[random_indices]
        y = tgt_z[random_indices]
    else:
        x = src_z
        y = tgt_z
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    kernels = []
    for sigma in sigmas:
        kernels.append(GaussianKernel(sigma=sigma))
    attr_mmd = 0.
    mmd = MultipleKernelMaximumMeanDiscrepancy(kernels=kernels)
    attr_mmd += mmd(x,y)
    return attr_mmd

def simple_mmd_kernel(source, target):
    source = torch.mean(source, dim=0)
    target = torch.mean(target, dim=0)
    return torch.exp(-0.1*torch.norm(source - target))

