import sys
sys.path.append('../..')
import argparse
import torch
import numpy as np
from model import MSGDA
from dataloader import MixMultiSrcAndTarLoader
from datetime import datetime, timezone, timedelta
import time
from torch_geometric.loader import DataLoader
from itertools import cycle, chain
from utils import *
from tqdm import tqdm
import os
import time

parser = argparse.ArgumentParser(description='Multi-Source Graph Domain Adaptation')
parser.add_argument('--tgt_name', type=str, default='0', help='target domain')
parser.add_argument('--src_name', type=str, default='4,1', help='source domain')
parser.add_argument('--construct_mode', type=str, default='emb', help='[emb, emb_homo] how to build XSrcD Graph')
parser.add_argument('--edge_feat', type=str, default='graphlet', help='cross-source-domain edge feature, like "emb,feat,graphlet" means edge_dim=3 with 3 kinds of edge feature')
parser.add_argument('--edge_norm', type=int, default=1, help='whether to normalize edge attr')
parser.add_argument('--encoder', type=str, default='gat', help='[gcn gat] for cross-source-domain MP')
parser.add_argument('--cross_output', type=str, default='concat', help='[naive, concat] how to combine original embedding and cross-graph embedding')
parser.add_argument('--s_pair_l', type=float, default=1, help='all s-s pair mmd loss')
parser.add_argument('--weighting', type=str, default='disc_exp', help="[disc_exp, disc_log, graphlet] weighting source domain, d=discriminator, g=graphlet")
parser.add_argument('--refine', type=str, default='raw', help='[raw, emb] use what to refine target logits')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epoch', type=int, default=1000, help='epochs')
parser.add_argument('--lambda_b', type=float, default=1)
parser.add_argument('--lambda_gp', type=int, default=5)
parser.add_argument('--critic_loop', type=int, default=10)
parser.add_argument('--cuda', type=int, default=0, help='cuda id')
parser.add_argument('--repeat', type=int, default=3,help='repear trainging times')
parser.add_argument('--no_up', type=int, default=300)
parser.add_argument('--knn', type=int, default=5)
args = parser.parse_args()
# HyperParams related to TwiBot
feat_dim = 10
num_classes = 2
encoder_dim = 128
hidden_dim = 128
CRITIC_ITERATIONS = args.critic_loop

def get_time():
    return datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d_%H-%M')

if not os.path.exists("log"):
    os.mkdir("log")
logfile = f"log/{get_time()}_src{args.src_name}_tgt{args.tgt_name}.log"


def evaluate(data):
    out,nbr_sim = model.inference(data, args.refine)
    metric, plog = calc_metrics(y=data.y, pred=out)
    return metric, plog



if __name__ == '__main__':
    coeff = {"LAMBDA":args.lambda_b,"LAMBDA_GP":args.lambda_gp}
    args.device = torch.device("cuda:{}".format(args.cuda) if (torch.cuda.is_available() and args.cuda>=0) else "cpu")
    src_idxs = args.src_name.split(',')
    src_idx_map = {idx:src_idxs[idx] for idx in range(len(src_idxs))}
    src_tgt_idx_loader, src_tgt_data = MixMultiSrcAndTarLoader(tgt_idx=args.tgt_name, src_idx=src_idxs, src_batch_size=args.batch_size//len(src_idxs), tgt_batch_size=args.batch_size, device=args.device)

    f1   = AverageMeter('F1', ':6.2f')
    auc  = AverageMeter('auc', ':6.2f')
    ap   = AverageMeter('ap', ':6.2f')

    for r in range(args.repeat):
        best_f1 = 0.
        best_auc = 0.
        best_ap = 0.

        model = MSGDA(encoder_type=args.encoder, encoder_dim=encoder_dim, hidden_dim=hidden_dim,num_classes=num_classes, device=args.device, dropout=args.dropout,coeff=coeff, cross_output=args.cross_output, src_pair_loss=args.s_pair_l, edge_feat=args.edge_feat, edge_norm=bool(args.edge_norm), construct_mode=args.construct_mode, weighting=args.weighting, knn=args.knn).to(args.device)
        optimizer = torch.optim.Adam(params=chain(model.encoder.parameters(), model.cls_model.parameters(), model.crossProp.parameters()), lr=args.lr, weight_decay=args.reg)
        optimizer_critic = torch.optim.Adam(params=chain(model.discriminator.parameters()), lr=args.lr,
                                            weight_decay=args.reg)
        
        pbar = tqdm(range(args.epoch), ncols=0)
        f1_no_up_cnt = 0
        start_time = time.time()
        for epoch in pbar:
            model.train()
            total_loss = 0.
            total_clf_loss = 0.
            total_d_loss = 0.
            for idx, src_tgt_data_idx in enumerate(zip(*src_tgt_idx_loader)):  # Iterate in batches over the training dataset.
                src_idx_list = [data[0].to(args.device) for data in src_tgt_data_idx[:-1]]
                tgt_idx      = src_tgt_data_idx[-1][0].to(args.device)
                CRITIC_ITERATIONS = max(CRITIC_ITERATIONS-epoch//2,1)
                for inner_iter in range(CRITIC_ITERATIONS):
                    critic_loss = model.forward_critic(src_idx_list, src_tgt_data[:-1], tgt_idx, src_tgt_data[-1])
                    optimizer_critic.zero_grad()
                    critic_loss.backward(retain_graph=True)
                    optimizer_critic.step()
                    #print('critic',critic_loss)

                loss,clf_loss,d_loss = model(src_idx_list, src_tgt_data[:-1], tgt_idx, src_tgt_data[-1])  # Perform a single forward pass.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
                total_loss += loss
                total_clf_loss += clf_loss
                total_d_loss += d_loss
            epoch_loss = total_loss / (idx + 1)
            epoch_clf_loss = total_clf_loss /(idx+1)
            epoch_d_loss = total_d_loss / (idx+1)
            if epoch % 1 == 0:
                model.eval()
                all_src_plog = ""
                # evaluate source
                for graph_idx in range(len(src_idxs)):
                    src_metric, src_plog = evaluate(src_tgt_data[graph_idx])
                    all_src_plog += (f" Source {src_idx_map[graph_idx]}: "+src_plog)
                # evaluate target
                tgt_metric, tgt_plog = evaluate(src_tgt_data[-1])
                print(f'Epoch: {epoch:03d}, loss:{epoch_loss}, {all_src_plog} Target:{tgt_plog}')

                if tgt_metric['f1-score']>best_f1:
                    best_f1  = tgt_metric['f1-score']
                    best_auc = tgt_metric['roc-auc']
                    best_ap  = tgt_metric['ap']
                    f1_no_up_cnt = 0
                else:
                    f1_no_up_cnt += 1
                with open(logfile, 'a+') as f:
                    f.write(f'Repeat:{r}, Epoch: {epoch:03d}, loss:{epoch_loss}, clf_loss:{epoch_clf_loss}, domain_loss:{epoch_d_loss}，{all_src_plog}, Target:{tgt_plog}, Best-Now:f1 {best_f1} auc {best_auc} ap {best_ap}\n')
                pbar.set_postfix_str('Target f1 {} :: Best f1 {} no up cnt {} '.format(tgt_metric['f1-score'], best_f1, f1_no_up_cnt))
            if epoch%20==0:
                end_time = time.time()
                with open(logfile, 'a+') as f:
                    f.write(f'The latest 20 epoch costs {end_time-start_time} seconds\n')
                start_time = time.time()

            if f1_no_up_cnt == args.no_up:
                break   
        f1.update(best_f1)
        auc.update(best_auc)
        ap.update(best_ap)
        model.zero_grad()
        with open(logfile, "a+") as f:
            f.write(
                f"PRE RUNS AVG RESULT ON Target Graph: F1:{round(f1.avg, 4)}/{round(f1.var, 4)}, AUC:{round(auc.avg, 4)}/{round(auc.var, 4)}, AP:{round(ap.avg, 4)}/{round(ap.var, 4)}\n")
            f.write(
                f"PRE RUNS BEST RESULT ON Target Graph: F1:{round(f1.max, 4)}, AUC:{round(auc.max, 4)}, AP:{round(ap.max, 4)}\n")

    with open(logfile,"a+") as f:
        f.write(f"FINAL RESULT ON Target Graph: F1:{round(f1.avg,4)}/{round(f1.var,4)}, AUC:{round(auc.avg,4)}/{round(auc.var,4)}, AP:{round(ap.avg,4)}/{round(ap.var,4)}")
                
