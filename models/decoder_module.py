

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class DecoderModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
    channels=256):
        super().__init__() 
        # From Proposal module
        self.num_class = num_class 
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = 100#num_proposal
        self.sampling = sampling
        self.score_out = 2+3+num_heading_bin*2+num_size_cluster*4+self.num_class # For out dim

        # self.layers = _get_clones(decoder_layer, num_layers)
        # self.num_layers = num_layers
        # self.norm = norm
        # self.return_intermediate = return_intermediate
        d_model, nhead, dim_feedforward, dropout, activation = 256, 8, 2048, 0.1, "relu"
        num_layers = 6
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.query_embed = nn.Embedding(self.num_proposal, d_model)

        # self.bbox_embed = MLP(d_model, d_model, 3+3, 3)
        # self.class_embed = nn.Linear(d_model, num_class)
        self.bbox_embed_full = nn.Conv1d(d_model, self.score_out, 1)
                

    def forward(self, data_dict):
        # query_embed is trainable
        memory = data_dict["memory"].permute(2,0,1)
        query_embed = self.query_embed.weight #torch.rand(100,2,256).cuda() # ( Dim, Batch, # decode slots)
        query_embed = query_embed.unsqueeze(1).repeat(1, memory.shape[1], 1)
        output = torch.zeros_like(query_embed).cuda()

        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, query_pos=query_embed)
            
            intermediate.append(self.norm(output))
        intermediate = torch.stack(intermediate)
        
        data_dict["memory"]       = memory.permute(1, 2, 0)
        hs = data_dict["hidden_state"] = intermediate.transpose(1, 2)

        # outputs_class = self.class_embed(hs)
        # outputs_bbox = self.bbox_embed(hs)#.sigmoid()
        # outputs_class = self.class_embed(hs)
        outputs_bbox = self.bbox_embed_full(hs[-1].transpose(1,2))

        data_dict["selfAttn_features"] = outputs_bbox#[-1]
        # data_dict = self.decode_scores_new(outputs_bbox[-1], outputs_class[-1], data_dict)
        data_dict = self.decode_scores(outputs_bbox, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        return data_dict

    def decode_scores_new(self, outputs_bbox, outputs_class, data_dict):
        center = outputs_bbox[:,:,0:3]
        size = outputs_bbox[:,:,3:6]
        sem_cls_scores = outputs_class

        data_dict['center'] = center
        data_dict['size']   = size
        data_dict['sem_cls_scores'] = sem_cls_scores
        return data_dict

    def decode_scores(self, net, data_dict, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
        """
        decode the predicted parameters for the bounding boxes

        """
        net_transposed = net.transpose(2,1).contiguous() # (batch_size, 1024, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        objectness_scores = net_transposed[:,:,0:2]

        # base_xyz = data_dict['sa4_xyz'] # (batch_size, num_proposal, 3)
        center = net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)

        heading_scores = net_transposed[:,:,5:5+num_heading_bin]
        heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
        
        size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
        
        sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10

        # store
        data_dict['objectness_scores'] = objectness_scores
        data_dict['center'] = center
        data_dict['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
        data_dict['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
        data_dict['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin
        data_dict['size_scores'] = size_scores
        data_dict['size_residuals_normalized'] = size_residuals_normalized
        data_dict['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        data_dict['sem_cls_scores'] = sem_cls_scores

        return data_dict

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=memory,
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
       
        return tgt

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


if __name__=='__main__':
    net = VotingModule(2, 256).cuda()
    xyz, features = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda())
    print('xyz', xyz.shape)
    print('features', features.shape)
