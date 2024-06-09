import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.function as fn



def neighbor_index(pos: torch.Tensor, rcut, box_size, device='cpu'):
    pos = pos.to(device)
    rel_pos = pos[None, :, :] - pos[:, None, :]
    rel_pos = torch.remainder(rel_pos + 0.5 * box_size, box_size) - 0.5 * box_size
    distance = torch.norm(rel_pos, dim=2)

    n = pos.size(0)
    idx_matrix = torch.arange(n).view(-1, 1).repeat(1, n).view(-1).to(device)
    
    mask = (distance.view(-1) <= rcut) & (idx_matrix != idx_matrix.view(n, n).t().reshape(-1))
    
    edge_idx_1 = idx_matrix[mask]
    edge_idx_2 = idx_matrix.view(n, n).t().reshape(-1)[mask]
    
    edge_idx = torch.stack([edge_idx_1, edge_idx_2], dim=0)
    return edge_idx


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, n_hidden=3, activation_first=False):
        super(MLP, self).__init__()
        act_fn = nn.ReLU()
        mlp_layers = []
        if activation_first:
            mlp_layers += [act_fn, nn.Linear(in_dim, hidden_dim), act_fn]
        else:
            mlp_layers += [nn.Linear(in_dim, hidden_dim), act_fn]
        for _ in range(1, n_hidden):
            mlp_layers += [nn.Linear(hidden_dim, hidden_dim), act_fn]
        mlp_layers += [nn.Linear(hidden_dim, out_dim)]
        self.mlp_layer = nn.Sequential(*mlp_layers)

    def forward(self, input):
        return self.mlp_layer(input)

class GNNLayer(nn.Module):
    def __init__(self, in_node_feat, in_edge_feat, out_node_feats, 
                 hidden_dim=128):
        super(GNNLayer, self).__init__()
        self.src_node_affine_function = nn.Linear(in_node_feat, hidden_dim)
        self.dst_node_affine_function = nn.Linear(in_node_feat, hidden_dim)
        self.phi_function = MLP(hidden_dim, in_node_feat, hidden_dim=hidden_dim, activation_first=True, n_hidden=2)
        self.theta_function = MLP(hidden_dim, out_node_feats, activation_first=True, n_hidden=1)
        self.phi_dst_node = nn.Linear(in_node_feat, hidden_dim)
        self.phi_updated_node = nn.Linear(in_node_feat, hidden_dim)

    def forward(self, g: dgl.DGLGraph, node_feat: torch.Tensor) -> torch.Tensor:
        h = node_feat.clone()
        with g.local_scope():
            g.srcdata['h'] = h
            g.dstdata['h'] = h
            edge_idx = g.edges()
            src_idx = edge_idx[0]
            dst_idx = edge_idx[1]
            encoded_edge_feat = g.edata['e']
            src_node_code = self.src_node_affine_function(h[src_idx])
            dst_node_code = self.dst_node_affine_function(h[dst_idx])
            temp_edge_code = encoded_edge_feat + src_node_code + dst_node_code
            g.edata['e_emb'] = self.phi_function(temp_edge_code) 
            g.update_all(fn.u_mul_e('h', 'e_emb', 'm'), fn.sum('m', 'h'))
            updated_node_emb = g.ndata['h']
            node_feat = self.theta_function(self.phi_dst_node(h) + self.phi_updated_node(updated_node_emb))
            return node_feat
        

class GNNBlock(nn.Module):
    def __init__(self, in_node_feat, out_node_feat, hidden_dim=128, n_conv_layers=4, edge_embed_dim=128):
        super(GNNBlock, self).__init__()
        self.gnn = nn.ModuleList()
        self.edge_embed_dim = edge_embed_dim

        for layer in range(n_conv_layers):
            if layer == 0:
                self.gnn.append(GNNLayer(in_node_feat=in_node_feat, in_edge_feat=self.edge_embed_dim, out_node_feats=out_node_feat, 
                                         hidden_dim=hidden_dim))
            else:
                self.gnn.append(GNNLayer(in_node_feat=out_node_feat, in_edge_feat=self.edge_embed_dim, out_node_feats=out_node_feat, 
                                         hidden_dim=hidden_dim))
                
    def forward(self, h: torch.Tensor, graph: dgl.DGLGraph) -> torch.Tensor:

        for l, gnn_layer in enumerate(self.gnn):
            h = h + gnn_layer.forward(graph, h)

        return h 


class MDNet(nn.Module):
    def __init__(self, encoding_size, out_feat, box_size, cutoff, hidden_dim=128, gnn_layers=4, 
                 edge_embedding_dim=128):
        super(MDNet, self).__init__()
        self.graph_nn = GNNBlock(in_node_feat=encoding_size, out_node_feat=encoding_size, 
                                 hidden_dim=hidden_dim, n_conv_layers=gnn_layers, 
                                 edge_embed_dim=edge_embedding_dim)
        self.edge_embed_dim = edge_embedding_dim
        self.box_size = box_size
        self.cutoff = cutoff
        self.node_embed = nn.Parameter(torch.randn((1, encoding_size)), requires_grad=True)
        self.edge_encoder = MLP(4, self.edge_embed_dim, hidden_dim=hidden_dim)
        self.graph_decoder = MLP(encoding_size, out_feat, n_hidden=2, hidden_dim=hidden_dim)


    def calc_bare_edge_feat(self, src_idx: torch.Tensor, dst_idx: torch.Tensor, 
                            pos: torch.Tensor):
        rel_pos = pos[dst_idx.long()] - pos[src_idx.long()]
        rel_pos_periodic = torch.remainder(rel_pos + 0.5 * self.box_size, 
                                           self.box_size) - 0.5 * self.box_size
        rel_pos_norm = rel_pos_periodic.norm(dim=1).view(-1, 1)
        rel_pos_periodic /= (rel_pos_norm + 1e-8)

        bare_edge_feat = torch.cat((rel_pos_periodic, rel_pos_norm), dim=1)
        
        return bare_edge_feat
    
    def build_graph(self, pos: torch.Tensor, edge_idx: torch.Tensor, self_loop=False) -> dgl.DGLGraph:
        center_idx = edge_idx[0, :]
        neigh_idx = edge_idx[1, :]
        center_idx = center_idx.long()
        neigh_idx = neigh_idx.long()
        graph = dgl.graph((center_idx, neigh_idx))
        graph = graph.to(device)
        bare_edge_feat = self.calc_bare_edge_feat(center_idx, neigh_idx, pos)
        embedded_edge_feat = self.edge_encoder(bare_edge_feat)
        graph.edata['e'] = embedded_edge_feat

        if self_loop:
            graph.add_self_loop()
        return graph 
    
    def build_graph_batches(self, pos_lst, edge_idx_lst):
        graph_lst = []
        for pos, edge_idx in zip(pos_lst, edge_idx_lst):
            graph = self.build_graph(pos, edge_idx)
            graph_lst += [graph]
        batched_graph = dgl.batch(graph_lst)
        return batched_graph
    
    def forward(self, pos_lst,  edge_idx_lst):
        b = len(pos_lst)
        N = pos_lst[0].shape[0]
        if b > 1:
            graph_batch = self.build_graph_batches(pos_lst, edge_idx_lst)
        else:
            graph_batch = self.build_graph(pos_lst[0], edge_idx_lst[0])
        #graph_batch = graph_batch.to(device)
        num = b * N
        x = self.node_embed.repeat((num, 1))
        x = self.graph_nn(x, graph_batch)

        x = self.graph_decoder(x)
        return x
    
    def predict_forces(self, pos: torch.Tensor):

        edge_idx_tsr = neighbor_index(pos, self.cutoff, self.box_size)

        pred = self.forward([pos], [edge_idx_tsr])
        return pred
    
    def training_step(self, pos_lst, forces_lst, criterion, lamb=1e-3):
        batch_size = len(pos_lst)
        edge_idx_lst = []
        for i in range(batch_size):
            edge_idx = neighbor_index(pos_lst[i], self.cutoff, self.box_size)
            edge_idx_lst += [edge_idx]
            
        forces_pred = self.forward(pos_lst, edge_idx_lst)
        forces_gt = torch.cat(forces_lst, dim=0)

        loss = criterion(forces_pred, forces_gt)
        conservative_loss = (torch.mean(forces_pred)).abs()
        loss = loss + lamb * conservative_loss

        return loss