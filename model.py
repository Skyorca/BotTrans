import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch.autograd import Variable
from utils import compute_dist, simple_mmd_kernel
from torch_geometric.utils import to_undirected
from torch_geometric.nn import MessagePassing

class BotGATEncoder(nn.Module):
    def __init__(self, hidden_dim, num_prop_size=7, cat_prop_size=3, dropout=0.3):
        super(BotGATEncoder, self).__init__()
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 2),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 2),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim, 2)

        self.gnn1 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        self.gnn2 = GATConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(num_features=hidden_dim)


    def forward(self, num_prop, cat_prop, edge_index, edge_type=None):
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((n, c), dim=1)
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.bn(x)
        x = self.gnn1(x, edge_index)
        x = self.dropout(self.relu(x))
        #x = self.bn(x)
        x = self.gnn2(x, edge_index)
        return x

class BotGCNEncoder(nn.Module):
    def __init__(self, hidden_dim, num_prop_size=7, cat_prop_size=3, dropout=0.3):
        super(BotGCNEncoder, self).__init__()
        self.num_prop_size = num_prop_size
        self.cat_prop_size = cat_prop_size
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 2)
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 2)
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.gnn1 = GCNConv(hidden_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=hidden_dim)

    def forward(self, input, edge_index, edge_type=None):
        num_prop = input[:,:self.num_prop_size]
        cat_prop = input[:,self.num_prop_size:]
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((n, c), dim=1)
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.bn(x)
        x = self.gnn1(x, edge_index)
        x = self.dropout(self.relu(x))
        x = self.gnn2(x, edge_index)
        return x

class Attention(nn.Module):
    """attention of combining a list of embeddings"""
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs

class CrossSourceDomainMP(nn.Module):
    """
    construct cross-source-domain mixup graph Gmix
    propagate on Gmix and update node embeddings
    """
    def __init__(self,hidden_dim, encoder, cross_output, construct_mode='emb', knn=5, edge_feat="", edge_norm=True, src_pair_loss = 1):
        super(CrossSourceDomainMP, self).__init__()
        self.construct_mode = construct_mode
        self.edge_norm = edge_norm
        print(edge_norm)
        self.knn = knn
        self.encoder = encoder
        self.cross_output = cross_output
        self.edge_feat = list(filter(None,edge_feat.split(",")))  # maximum with [emb, feat, graphlet, label]
        self.edge_dim  = len(self.edge_feat)
        print(f"Constructing cross-source-domain graph with {self.edge_dim} types of edge_feat {self.edge_feat}")
        self.src_pair_loss = src_pair_loss
        if encoder=="gcn":
            self.gnn = GCNConv(hidden_dim,hidden_dim)
        elif encoder=="gat":
            self.gnn = GATConv(hidden_dim, hidden_dim//4, heads=4, edge_dim=self.edge_dim)

        self.output = Attention(hidden_dim) if cross_output!="naive" else None

    def compute_adj(self,src_v, construct_type):
        if construct_type=="label" :
            src_v = [v.reshape(-1,1) for v in src_v]
            v = torch.vstack(src_v)
            adj = torch.eq(v, v.T).float()
        else:
            v = torch.vstack(src_v)  # vstack across all src domains
            v = F.normalize(v, dim=1)
            adj = v @ v.T
        return adj

    def construct(self, src_feature:list, src_embedding:list, src_y:list,src_graphlet:list):
        n_node_list = [x.shape[0] for x in src_feature]
        self.n_node_list = n_node_list
        all_sim_mat = {"emb":None,"feat":None,"graphlet":None,"label":None}
        all_src_info = {"emb":src_embedding,"feat":src_feature,"graphlet":src_graphlet,"label":src_y}
        with torch.no_grad():
            # 1. compute all-source similarity matrix
            if self.construct_mode=="emb":
                adj = self.compute_adj(src_embedding,self.construct_mode)
                all_sim_mat[self.construct_mode] = adj
            if self.construct_mode=="emb_homo":
                adj = self.compute_adj(src_embedding,self.construct_mode)
                homo_mask = self.compute_adj(all_src_info["label"], "label")
                adj = adj * homo_mask

            # set block-diag to 0, only allow for cross-domain edges
            left_shift = 0
            right_shift = n_node_list[0]
            for i in range(len(n_node_list)):
                adj[left_shift:right_shift,left_shift:right_shift] = 0
                if i<len(n_node_list)-1:
                    left_shift += n_node_list[i]
                    right_shift += n_node_list[i+1]
            # 2 sparsification by knn
            _,topk_idx_2 = adj.topk(self.knn, dim=1)
            topk_idx_1 = torch.arange(0, sum(n_node_list)).reshape(sum(n_node_list), 1).repeat(1, self.knn)
            mask = torch.zeros_like(adj)
            mask[topk_idx_1, topk_idx_2] = 1.
            adj = adj*mask
            # 3 to edge_index
            sp_row, sp_col = torch.nonzero(adj,as_tuple=True)
            edge_index = torch.vstack([sp_row,sp_col])
            # 4 assign edge weight
            all_edge_attr = []
            for feat_type in self.edge_feat:
                mat = all_sim_mat[feat_type]
                if mat is None:
                    mat = self.compute_adj(all_src_info[feat_type], feat_type)
                # neighborhood-normalization
                if self.edge_norm:
                    norm_submat = F.softmax(mat[topk_idx_1, topk_idx_2],dim=1)
                    mat[topk_idx_1, topk_idx_2] = norm_submat
                sp_val = mat[sp_row,sp_col]
                all_edge_attr.append(sp_val.reshape(-1,1))
            edge_attr = torch.hstack(all_edge_attr) if len(all_edge_attr)>0 else None
        return edge_index, edge_attr

    def forward(self,src_feature:list,src_embedding:list,src_y:list,src_graphlet:list):
        """Three source domain properties define how a cross-source-graph are built and propagated"""
        loss = 0.
        edge_index, edge_attr = self.construct(src_feature=src_feature, src_embedding=src_embedding, src_y=src_y, src_graphlet=src_graphlet)
        emb = torch.vstack(src_embedding)
        if self.encoder=="gcn":
            # for GCN, not use edge weight
            updated_emb = self.gnn(x=emb, edge_index=edge_index)
        elif self.encoder=="gat":
            if self.edge_dim>0:
                updated_emb = self.gnn(x=emb, edge_index=edge_index, edge_attr=edge_attr)
            else:
                updated_emb = self.gnn(x=emb, edge_index=edge_index)
        if self.cross_output=="concat":
            updated_emb = self.output([emb,updated_emb])
        if self.src_pair_loss>0:
            emb_list = []
            for i in range(len(self.n_node_list)):
                emb_list.append(updated_emb[sum(self.n_node_list[:i]):sum(self.n_node_list[:i+1]),:])
            for i in range(len(self.n_node_list)):
                for j in range(len(self.n_node_list)):
                    if i==j: continue
                    #loss += compute_dist(emb_list[i],emb_list[j])
                    loss += simple_mmd_kernel(emb_list[i], emb_list[j])
            loss /= 2

        return loss, updated_emb

def weighting_d(x_ds,x_dt,disc,n_node_list,weighting_func):
    """Weighting each source domain with domain-level discrepancy with target"""
    w = []
    # grad or no_grad?
    with torch.no_grad():
        for i in range(len(n_node_list)):
            critic_ds = disc(x_ds[sum(n_node_list[:i]):sum(n_node_list[:i+1])]).reshape(-1)
            critic_dt = disc(x_dt).reshape(-1)
            discrepancy = torch.abs(torch.mean(critic_ds) - torch.mean(critic_dt))
            if "log" in weighting_func:
                w.append(-torch.log(discrepancy+1e-7))
            elif "exp" in weighting_func:
                w.append(torch.exp(-discrepancy))
        w = torch.softmax(torch.FloatTensor(w),dim=-1)
    return w

def weighting_g(g_ds,g_dt,n_node_list):
    w = []
    g_sim_mat = F.normalize(g_ds,dim=1)@F.normalize(g_dt,dim=1).T
    for i in range(len(n_node_list)):
        sim = torch.mean(g_sim_mat[sum(n_node_list[:i]):sum(n_node_list[:i+1]),:])
        w.append(sim)
    w = torch.softmax(torch.FloatTensor(w), dim=-1)
    return w

class NeighborAggr(MessagePassing):
    def __init__(self):
        super().__init__(aggr="mean")
    def forward(self,x,edge_index):
        edge_index = to_undirected(edge_index)
        out = self.propagate(edge_index, x=x)
        return out

class MSGDA(nn.Module):
    """
    MULTI SOURCE GRAPH DOMAIN ADAPTATION MODEL
    """
    def __init__(self,encoder_type, encoder_dim, hidden_dim, num_classes, device, dropout,coeff,cross_output, edge_feat, weighting, edge_norm=True, construct_mode="emb",src_pair_loss=0,knn=5):
        super(MSGDA, self).__init__()
        self.coeff = coeff
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device
        self.encoder_dim = encoder_dim
        self.src_pair_loss_w = src_pair_loss
        self.weighting = weighting
        self.encoder = BotGCNEncoder(hidden_dim=hidden_dim)

        self.cls_model = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(self.encoder_dim, 1),
            nn.Sigmoid()
        )
        self.criterion = nn.CrossEntropyLoss()
        self.crossProp = CrossSourceDomainMP(hidden_dim=hidden_dim, encoder=encoder_type, cross_output=cross_output, src_pair_loss=src_pair_loss, edge_feat=edge_feat, edge_norm=edge_norm, construct_mode=construct_mode, knn=knn)
        self.Aggr = NeighborAggr()
    def forward_critic(self, src_idx_list, src_data_list, tgt_idx, tgt_data):
        self.n_node_s = [len(x) for x in src_idx_list]
        # overall critic loss
        overall_critic_loss = 0.
        # naive computation between each source-target pair
        features_t = tgt_data.x
        edge_index_t = tgt_data.edge_index
        all_src_feat = []
        all_src_emb  = []
        all_src_y = []
        all_src_graphlet = []
        for src_graph in range(len(src_idx_list)):
            # full graph MP
            features_s = src_data_list[src_graph].x
            edge_index_s = src_data_list[src_graph].edge_index
            # only get the embeddings of batch nodes
            x_ds = self.encoder(features_s, edge_index_s)[src_idx_list[src_graph]]
            all_src_feat.append(features_s[src_idx_list[src_graph]])
            all_src_emb.append(x_ds)
            all_src_y.append((src_data_list[src_graph].y[src_idx_list[src_graph]]).float())
            all_src_graphlet.append(src_data_list[src_graph].x_graphlet[src_idx_list[src_graph]])
        _, updated_x_ds = self.crossProp(src_feature=all_src_feat, src_embedding=all_src_emb, src_y = all_src_y, src_graphlet=all_src_graphlet)
        x_dt = self.encoder(features_t, edge_index_t)[tgt_idx]
        # weighting
        if "disc" in self.weighting:
            w = weighting_d(updated_x_ds, x_dt,self.discriminator,self.n_node_s,self.weighting)
        elif self.weighting=="graphlet":
            w = weighting_g(torch.vstack(all_src_graphlet),tgt_data.x_graphlet,self.n_node_s)
        else:
            raise NotImplementedError
        for i in range(len(self.n_node_s)):
            critic_ds = self.discriminator(updated_x_ds[sum(self.n_node_s[:i]):sum(self.n_node_s[:i+1])]).reshape(-1)
            critic_dt = self.discriminator(x_dt).reshape(-1)
            gp = gradient_penalty(self.discriminator, updated_x_ds[sum(self.n_node_s[:i]):sum(self.n_node_s[:i+1])], x_dt, self.device)
            loss_critic = (
                    -torch.abs(torch.mean(critic_ds) - torch.mean(critic_dt)) + self.coeff['LAMBDA_GP'] * gp
            )
            overall_critic_loss += w[i]*loss_critic
        return overall_critic_loss

    def forward(self, src_idx_list, src_data_list, tgt_idx, tgt_data):
        features_t = tgt_data.x
        edge_index_t = tgt_data.edge_index
        all_src_feat = []
        all_src_emb  = []
        all_src_label = []
        all_src_graphlet = []

        for src_graph in range(len(src_idx_list)):
            # full graph MP
            features_s = src_data_list[src_graph].x
            edge_index_s = src_data_list[src_graph].edge_index
            # only get batch node embeddings/labels
            labels_s = src_data_list[src_graph].y[src_idx_list[src_graph]]
            x_ds = self.encoder(features_s, edge_index_s)[src_idx_list[src_graph]]
            all_src_feat.append(features_s[src_idx_list[src_graph]])
            all_src_emb.append(x_ds)
            all_src_label.append(labels_s.float())
            all_src_graphlet.append(src_data_list[src_graph].x_graphlet[src_idx_list[src_graph]])

        src_pair_loss, updated_x_ds = self.crossProp(src_feature=all_src_feat, src_embedding=all_src_emb, src_y = all_src_label, src_graphlet=all_src_graphlet)
        x_dt = self.encoder(features_t, edge_index_t)[tgt_idx]
        domain_loss = 0
        clf_loss = 0
        # weighting
        if "disc" in self.weighting:
            w = weighting_d(updated_x_ds, x_dt,self.discriminator,self.n_node_s,self.weighting)
        elif self.weighting=="graphlet":
            w = weighting_g(torch.vstack(all_src_graphlet),tgt_data.x_graphlet,self.n_node_s)
        else:
            raise NotImplementedError

        for i in range(len(self.n_node_s)):
            critic_ds = self.discriminator(updated_x_ds[sum(self.n_node_s[:i]):sum(self.n_node_s[:i+1])]).reshape(-1)
            critic_dt = self.discriminator(x_dt).reshape(-1)
            loss_critic = torch.abs(torch.mean(critic_ds) - torch.mean(critic_dt))
            domain_loss += w[i]*loss_critic
            src_logits = self.cls_model(updated_x_ds[sum(self.n_node_s[:i]):sum(self.n_node_s[:i+1])])
            clf_loss    += w[i]*self.criterion(src_logits, all_src_label[i].long())
        total_loss = clf_loss + self.coeff['LAMBDA'] * domain_loss+self.src_pair_loss_w*src_pair_loss
        return total_loss, clf_loss, domain_loss

    def inference(self, data, refine_func):
        x, edge_index = data.x, data.edge_index
        emb = self.encoder(x, edge_index)
        logits = self.cls_model(emb)
        if refine_func=="raw":
            x_nbr = self.Aggr(x, edge_index)
            nbr_sim = torch.sum(F.normalize(x) * F.normalize(x_nbr), dim=1)
        elif refine_func=="emb":
            x_nbr = self.Aggr(emb, edge_index)
            nbr_sim = torch.sum(F.normalize(emb) * F.normalize(x_nbr), dim=1)
        else:
            raise  NotImplementedError
        logits[:,-1] = 0.5*logits[:,-1]+0.5*(1-nbr_sim)
        return logits, nbr_sim


def gradient_penalty(critic, src, tgt, device):
    Nodes_num = min(src.shape[0], tgt.shape[0])  # 梯度惩罚时的结点数
    features = src.shape[1]

    # create the interpolated nodes
    # alpha = torch.rand((Nodes_num, 1)).repeat(1,features).to(device)
    # random_points = alpha*(src[:Nodes_num]) + ((1 - alpha)*(tgt[:Nodes_num]))

    # Calculate critic scores
    # mixed_scores = critic(random_points)
    inputs = torch.vstack([src, tgt])
    scores = critic(inputs)
    # # Take the gradient of the scores with respect to the image
    gradient = torch.autograd.grad(
        inputs=inputs,
        outputs=scores,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)  # L2 norm
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    # print(gradient_penalty)
    return gradient_penalty


def to_onehot(label_matrix, num_classes, device):
    identity = torch.eye(num_classes).to(device)
    onehot = torch.index_select(identity, 0, label_matrix)
    return onehot
