# Modified version such that it works with 3D scence boxes
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from data.scannet.model_util_scannet import ScannetDatasetConfig
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch,generalized_box_iou
import numpy as np
# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.DC = ScannetDatasetConfig()

    @torch.no_grad()
    # def forward(self, outputs, targets):
    def forward(self,data_dict):    
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        pred_corner,gt_corner,pred_center,gt_center = self.get_corner(data_dict) # List of batch box, length is batch size

        pred_logits = data_dict['sem_cls_scores']
        gt_logits = data_dict["sem_cls_label"]
        gt_num_bbox = data_dict["num_bbox"]

        bs, num_queries = pred_logits.shape[:2]
        # We flatten to compute the cost matrices in a batch
        out_prob = pred_logits.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = pred_corner.reshape(-1, pred_corner.shape[-2],pred_corner.shape[-1])  # [batch_size * num_queries, 3]

        tgt_ids = []
        for batch in range(bs):
            for item in range(gt_num_bbox[batch]):
                tgt_ids.append(gt_logits[batch][item])
        tgt_ids = torch.IntTensor(tgt_ids).long()

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids].cpu()

        # flat_out_bbox = out_bbox.reshape(out_bbox.shape[0],-1)
        # flat_out_bbox = torch.Tensor(flat_out_bbox)
        # flat_gt_corner = gt_corner.reshape(gt_corner.shape[0],-1)
        # flat_gt_corner = torch.Tensor(flat_gt_corner)

        # # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(flat_out_bbox, flat_gt_corner, p=1).cpu()

        # Use center loss to replace the box loss in Deter matcher
        pred_center = pred_center.reshape(-1,pred_center.shape[-1])
        pred_center = torch.Tensor(pred_center)
        gt_center   = torch.Tensor(gt_center)
        cost_bbox = torch.cdist(pred_center,gt_center, p=1).cpu()




        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(out_bbox,gt_corner)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = gt_num_bbox
        sizes = [gt_num_bbox[i] for i in range (gt_num_bbox.shape[0])]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        data_dict["match_indices_list"] = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return data_dict
        
    def get_corner(self,data):
        # predicted box
        pred_center = data['center'].detach().cpu().numpy()
        pred_heading_class = torch.argmax(data['heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
        pred_size_class = torch.argmax(data['size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(data['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class.detach().cpu().numpy()
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

        # ground truth bbox
        gt_center = data['center_label'].cpu().numpy() # (B,128,3)
        number_box = data["num_bbox"].cpu().numpy() #B
        gt_heading_class = data['heading_class_label'].cpu().numpy() # B,128
        gt_heading_residual = data['heading_residual_label'].cpu().numpy() # B,128
        gt_size_class = data['size_class_label'].cpu().numpy() # B,128
        gt_size_residual = data['size_residual_label'].cpu().numpy() # B,128,3


        pred_corner      = []
        gt_corner        = []
        pred_center_list = []
        gt_center_list   = []

        for i in range(pred_center.shape[0]):
            pred_obb_batch = self.DC.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                            pred_size_class[i], pred_size_residual[i])
            pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
            pred_corner.append(pred_bbox_batch)
            

            gt_obb_batch = self.DC.param2obb_batch(gt_center[i, :number_box[i], 0:3], gt_heading_class[i,:number_box[i]], gt_heading_residual[i,:number_box[i]],
                            gt_size_class[i,:number_box[i]], gt_size_residual[i,:number_box[i],:])
            gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
            gt_corner.append(gt_bbox_batch)
            pred_center_list.append(pred_center[i, :, 0:3])
            gt_center_list.append(gt_center[i, :number_box[i], 0:3])
        
        pred_corner = np.stack(pred_corner)
        gt_corner = np.vstack(gt_corner)
        pred_center_list = np.stack(pred_center_list)
        gt_center_list = np.vstack(gt_center_list)

        return pred_corner,gt_corner,pred_center_list,gt_center_list

def build_matcher():
    # return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

    return HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
