

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EncoderModule(nn.Module):
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

        # Self attention part
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sAttn1 = SA_Layer(channels)
        self.sAttn2 = SA_Layer(channels)
        self.sAttn3 = SA_Layer(channels)
        self.sAttn4 = SA_Layer(channels)
        
        self.conv_fuse = nn.Sequential(nn.Conv1d(channels*5, channels*4, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(channels*4),
                                   nn.LeakyReLU(0.2))
        
        self.convs1 = nn.Conv1d(channels*4*3, channels*2, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)

        self.convs2 = nn.Conv1d(channels*2, channels, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        
        # self.convs3 = nn.Conv1d(channels, channels+100, 1)
        self.bns3 = nn.BatchNorm1d(256)   

        # self.conv3 = nn.Conv1d(channels, self.score_out, 1)

        self.convs_test = nn.Conv1d(channels, channels, 1)
        self.bns_test   = nn.BatchNorm1d(channels)
        self.dp_test    = nn.Dropout(0.5)
        
    def forward(self, data_dict):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape

        x = data_dict['sa4_features']
        batch_size, _, N = x.size()

        # B, D, N
        # Self attention + pre norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sAttn1(x)
        # x2 = self.sAttn2(x1)
        # x3 = self.sAttn3(x2)
        # x4 = self.sAttn4(x3)

        # Fuse output of each four attention layer
        # x = torch.cat((x, x1, x2, x3, x4), dim=1)
        # x = self.conv_fuse(x)
        
        # x_max = torch.max(x, 2)[0]
        # x_avg = torch.mean(x, 2)
        # x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        # x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        # x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1) # 1024 
        # x = torch.cat((x, x_global_feature), 1) # 1024 * 3 

        # #Feed forward
        # x = F.leaky_relu(self.bns1(self.convs1(x)), negative_slope=0.2)
        # x = x + self.dp1(x)
        # x = F.leaky_relu(self.bns2(self.convs2(x)), negative_slope=0.2)
        # x = x + self.dp2(x)
        # x = self.bns3(x)

        x = x = F.leaky_relu(self.bns_test(self.convs_test(x1)), negative_slope=0.2)
        x = x + self.dp_test(x)

        data_dict["memory"] = x
        
        # data_dict["selfAttn_features"] = x
        # x = self.conv3(x)
        # data_dict = self.decode_scores(x, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)


        return data_dict

    def decode_scores(self, net, data_dict, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
        """
        decode the predicted parameters for the bounding boxes

        """
        net_transposed = net.transpose(2,1).contiguous() # (batch_size, 1024, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        objectness_scores = net_transposed[:,:,0:2]

        base_xyz = data_dict['sa4_xyz'] # (batch_size, num_proposal, 3)
        center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)

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
