

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CombineModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
    channels=256):
        super().__init__() 
        # channels = 1024
        # From Proposal module
        self.num_class = num_class 
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = 100#num_proposal
        self.sampling = sampling

        self.sAttn1 = SA_Layer(channels)
        self.sAttn2 = SA_Layer(channels)
        self.sAttn3 = SA_Layer(channels)
        self.sAttn4 = SA_Layer(channels)

        self.conv_fuse = nn.Sequential(nn.Conv1d(channels*5, channels*4, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(channels*4),
                                   nn.LeakyReLU(0.2))
        
        self.convs1 = nn.Conv1d(channels*4*3, channels*2, 1)
        self.bns1 = nn.BatchNorm1d(channels*2)
        self.dp1 = nn.Dropout(0.5)

        self.convs2 = nn.Conv1d(channels*2, channels, 1)
        self.bns2 = nn.BatchNorm1d(channels)
        self.dp2 = nn.Dropout(0.5)
        
        self.bns3 = nn.BatchNorm1d(channels)

        hidden_size = channels
        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )
        
    def forward(self, data_dict):
        zeros = torch.zeros((data_dict["aggregated_vote_features"].shape[0],data_dict["aggregated_vote_features"].shape[1])).cuda()
        ones = torch.ones((data_dict["bert_out_hidden"].shape[0],data_dict["bert_out_hidden"].shape[1])).cuda()
        attention_mask = torch.cat([zeros,ones],dim=1)

        comebine = torch.cat([data_dict["aggregated_vote_features"],data_dict["bert_out_hidden"]],dim=1).transpose(2,1)  # B,Dim, Size


        x = comebine#.permute(1,0,2)        # Size, B, Dim
        batch_size, _, N = x.size()
        x1 = self.sAttn1(x)#,attention_mask)
        x2 = self.sAttn2(x1)#,attention_mask)
        x3 = self.sAttn3(x2)#,attention_mask)
        x4 = self.sAttn4(x3)#,attention_mask)

        x = torch.cat((x, x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1) # 1024 
        x = torch.cat((x, x_global_feature), 1) # 1024 * 3 

        #Feed forward
        x = F.leaky_relu(self.bns1(self.convs1(x)), negative_slope=0.2)
        x = x + self.dp1(x)
        x = F.leaky_relu(self.bns2(self.convs2(x)), negative_slope=0.2)
        x = x + self.dp2(x)
        x = self.bns3(x)

        data_dict["memory"] = x 
        combined_feature = x[:,:,:data_dict["aggregated_vote_features"].shape[1]]
        confidences = self.match(combined_feature).squeeze(1) # batch_size, num_proposals
                
        data_dict["cluster_ref"] = confidences

        return data_dict

    

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

        # nhead = 8
        dropout = 0.1
        # self.self_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

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

        # x_r = self.self_attn(x, x, value=x, attn_mask=src_mask)[0]
        x = x + self.dropout1(x_r)
        return x


if __name__=='__main__':
    net = VotingModule(2, 256).cuda()
    xyz, features = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda())
    print('xyz', xyz.shape)
    print('features', features.shape)
