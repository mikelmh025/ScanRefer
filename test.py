# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import sys
# import os

# # sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
# # from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

# # class Pointnet2Backbone(nn.Module):
# #     r"""
# #        Backbone network for point cloud feature learning.
# #        Based on Pointnet++ single-scale grouping network. 
        
# #        Parameters
# #        ----------
# #        input_feature_dim: int
# #             Number of input channels in the feature descriptor for each point.
# #             e.g. 3 for RGB.
# #     """
# #     def __init__(self, input_feature_dim=0,attn=False):
# #         super().__init__()

# #         self.input_feature_dim = input_feature_dim
# #         self.attn = attn

# #         # --------- 4 SET ABSTRACTION LAYERS ---------
# #         self.sa1 = PointnetSAModuleVotes(
# #                 npoint=2048,
# #                 radius=0.2,
# #                 nsample=64,
# #                 mlp=[input_feature_dim, 64, 64, 128],
# #                 use_xyz=True,
# #                 normalize_xyz=True
# #             )

# #         self.sa2 = PointnetSAModuleVotes(
# #                 npoint=1024,
# #                 radius=0.4,
# #                 nsample=32,
# #                 mlp=[128, 128, 128, 256],
# #                 use_xyz=True,
# #                 normalize_xyz=True
# #             )

# #         self.sa3 = PointnetSAModuleVotes(
# #                 npoint=512,
# #                 radius=0.8,
# #                 nsample=16,
# #                 mlp=[256, 128, 128, 256],
# #                 use_xyz=True,
# #                 normalize_xyz=True
# #             )

# #         self.sa4 = PointnetSAModuleVotes(
# #                 npoint=256,
# #                 radius=1.2,
# #                 nsample=16,
# #                 mlp=[256, 128, 128, 256],
# #                 use_xyz=True,
# #                 normalize_xyz=True
# #             )

# #         # --------- 2 FEATURE UPSAMPLING LAYERS --------
# #         self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
# #         self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

# #         self.multhead_attn = nn.MultiheadAttention(256,8)
# #         self.multhead_attn2 = nn.MultiheadAttention(256,8)

# #     def _break_up_pc(self, pc):
# #         xyz = pc[..., :3].contiguous()
# #         features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

# #         return xyz, features

# #     def forward(self, data_dict):
# #         r"""
# #             Forward pass of the network

# #             Parameters
# #             ----------
# #             pointcloud: Variable(torch.cuda.FloatTensor)
# #                 (B, N, 3 + input_feature_dim) tensor
# #                 Point cloud to run predicts on
# #                 Each point in the point-cloud MUST
# #                 be formated as (x, y, z, features...)

# #             Returns
# #             ----------
# #             data_dict: {XXX_xyz, XXX_features, XXX_inds}
# #                 XXX_xyz: float32 Tensor of shape (B,K,3)
# #                 XXX_features: float32 Tensor of shape (B,K,D)
# #                 XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
# #         """
# #         # import sys

# #         pointcloud = data_dict#["point_clouds"]
# #         data_dict = {}

# #         batch_size = pointcloud.shape[0]

# #         xyz, features = self._break_up_pc(pointcloud)

# #         # --------- 4 SET ABSTRACTION LAYERS ---------        
# #         xyz, features, fps_inds = self.sa1(xyz, features)
# #         data_dict['sa1_inds'] = fps_inds
# #         data_dict['sa1_xyz'] = xyz
# #         data_dict['sa1_features'] = features

# #         xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
# #         data_dict['sa2_inds'] = fps_inds
# #         data_dict['sa2_xyz'] = xyz
# #         data_dict['sa2_features'] = features

# #         xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
# #         data_dict['sa3_xyz'] = xyz
# #         data_dict['sa3_features'] = features

# #         xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
# #         data_dict['sa4_xyz'] = xyz
# #         data_dict['sa4_features'] = features




# #         # --------- 2 FEATURE UPSAMPLING LAYERS --------
# #         features = self.fp1(data_dict['sa3_xyz'], data_dict['sa4_xyz'], data_dict['sa3_features'], data_dict['sa4_features'])
# #         features = self.fp2(data_dict['sa2_xyz'], data_dict['sa3_xyz'], data_dict['sa2_features'], features)
    

# #         data_dict['fp2_features'] = features
# #         data_dict['fp2_xyz'] = data_dict['sa2_xyz']
# #         num_seed = data_dict['fp2_xyz'].shape[1]
# #         data_dict['fp2_inds'] = data_dict['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
# #         return data_dict

# # if __name__=='__main__':
# #     backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
# #     print(backbone_net)
# #     backbone_net.eval()
# #     out = backbone_net(torch.rand(16,20000,6).cuda())
# #     for key in sorted(out.keys()):
# #         print(key, '\t', out[key].shape)

# # from apex.normalization.fused_layer_norm import FusedLayerNorm
# # import apex
# # from apex.normalization.fused_layer_norm import FusedLayerNorm


# # input = torch.randn(20, 5, 10, 10)
# # m = FusedLayerNorm(input.size()[1:])
# # # With Learnable Parameters
# # m = apex.normalization.FusedLayerNorm(input.size()[1:])
# # # Without Learnable Parameters
# # m = apex.normalization.FusedLayerNorm(input.size()[1:], elementwise_affine=False)
# # # Normalize over last two dimensions
# # m = apex.normalization.FusedLayerNorm([10, 10])
# # # Normalize over last dimension of size 10
# # m = apex.normalization.FusedLayerNorm(10)
# # # Activating the module
# # output = m(input)

# # print(output.shape)


from __future__ import division
import scipy.optimize
import numpy as np

def bbox_iou(boxA, boxB):
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.
    
  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou



def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.

    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 


gt = np.array([[3, 1, 4, 4],[2, 6, 7, 7],[2, 2, 3, 3]])

pre = np.array([[1, 1, 2, 3],[2, 2, 4, 4],[1, 2, 3, 4]])
# print(gt[0])

iou = match_bboxes(gt,pre,IOU_THRESH=0.25)

print(iou)