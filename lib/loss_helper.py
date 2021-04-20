# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch,generalized_box_iou,accuracy, get_3d_box_batch_no_heading

from models.matcher import build_matcher

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8] # put larger weights on positive objectness

matcher = build_matcher()

def compute_vote_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = data_dict['seed_xyz'].shape[0]
    num_seed = data_dict['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = data_dict['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = data_dict['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(data_dict['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += data_dict['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict['center']
    # sa4_xyz
    gt_center = data_dict['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(data_dict, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = data_dict['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = data_dict['center']
    gt_center = data_dict['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = data_dict['box_label_mask']
    objectness_label = data_dict['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(data_dict['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(data_dict['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(data_dict['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(data_dict['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(data_dict['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(data_dict['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(data_dict['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(data_dict['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(data_dict['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def compute_reference_loss(data_dict, config,mask_aug):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """

    # unpack
    cluster_preds         = data_dict["cluster_ref"] # (B, num_proposal)
    cluster_ref_masked    = data_dict["cluster_ref_masked"] if mask_aug==True else None # (B, num_proposal)
    gt_neg_boxes          = data_dict["neg_boxes"].detach().cpu().numpy()  #(B, Added, 6) The feature size of each box is 6
    gt_neg_boxes          = np.zeros((gt_neg_boxes.shape[0], gt_neg_boxes.shape[1],gt_neg_boxes.shape[2]+1))
    gt_neg_boxes[:,:,0:6] = data_dict["neg_boxes"].detach().cpu().numpy() #(B, Added, 7) The feature size of each box is 6+1

    # predicted bbox
    pred_ref = data_dict['cluster_ref'].detach().cpu().numpy() # (B,num_proposal)
    pred_center = data_dict['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    # ground truth bbox
    gt_center = data_dict['ref_center_label'].cpu().numpy() # (B,3)
    gt_heading_class = data_dict['ref_heading_class_label'].cpu().numpy() # B
    gt_heading_residual = data_dict['ref_heading_residual_label'].cpu().numpy() # B
    gt_size_class = data_dict['ref_size_class_label'].cpu().numpy() # B
    gt_size_residual = data_dict['ref_size_residual_label'].cpu().numpy() # B,3
    # convert gt bbox parameters to bbox corners
    gt_obb_batch = config.param2obb_batch(gt_center[:, 0:3], gt_heading_class, gt_heading_residual,
                    gt_size_class, gt_size_residual)
    gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])

    # compute the iou score for all predictd positive ref
    batch_size, num_proposals = cluster_preds.shape
    labels = np.zeros((batch_size, num_proposals))
    label_contr = np.zeros((batch_size, 1+gt_neg_boxes.shape[1])) #(B, 1+neg)
    pred_contr = np.zeros((batch_size, 1+gt_neg_boxes.shape[1])) #(B, 1+neg)
    neg_iou_idx = np.zeros((batch_size, 1+gt_neg_boxes.shape[1]))

    ref_iou_idx = []
    for i in range(pred_ref.shape[0]):
        # convert the bbox parameters to bbox corners
        pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                    pred_size_class[i], pred_size_residual[i])
        pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
        ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[i], (num_proposals, 1, 1)))
        labels[i, ious.argmax()] = 1 # treat the bbox with highest iou score as the gt
        ref_iou_idx.append(ious.argmax())

        gt_neg_bbox_batch = get_3d_box_batch(gt_neg_boxes[i,:, 3:6], gt_neg_boxes[i,:, 6], gt_neg_boxes[i,:, 0:3])
        # find the negative sample boxes in prediction
        for neg_obj in range(gt_neg_boxes.shape[1]):
            
            neg_ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_neg_bbox_batch[neg_obj], (num_proposals, 1, 1)))
            pred_contr[i,neg_obj] = cluster_preds[i,neg_ious.argmax()]
            neg_iou_idx[i,neg_obj] = neg_ious.argmax()
            # label_contr[i,neg_obj] = 0
            # gt_neg_boxes[i,neg_obj]
        pred_contr[i,gt_neg_boxes.shape[1]] = cluster_preds[i,ious.argmax()]
        label_contr[i,gt_neg_boxes.shape[1]] = 1
        neg_iou_idx[i,gt_neg_boxes.shape[1]] = ious.argmax()

    label_contr = torch.FloatTensor(label_contr).cuda()
    pred_contr  = torch.FloatTensor(pred_contr).cuda()

    cluster_labels = torch.FloatTensor(labels).cuda()

    # reference loss
    criterion = SoftmaxRankingLoss()
    loss       = criterion(cluster_preds, cluster_labels.float().clone())
    loss_mask  = criterion(cluster_ref_masked, cluster_labels.float().clone()) if mask_aug==True else torch.zeros(1)[0].cuda()
    loss_contr = criterion(pred_contr,label_contr.float().clone())

    # # 1 vs all other objects with same label, across the entire batch 
    # all_gt_center_label             = data_dict['center_label'].cpu().numpy()           # (B,128,3)
    # all_gt_heading_class_label      = data_dict["heading_class_label"].cpu().numpy()    # (B,128,3)
    # all_gt_heading_residual_label   = data_dict["heading_residual_label"].cpu().numpy() # (B,128)
    # all_gt_size_class_label         = data_dict["size_class_label"].cpu().numpy()       # (B,128)
    # all_gt_size_residual_label      = data_dict["size_residual_label"].cpu().numpy()    # (B,128,3)

    # all_gt_sem_labels               = data_dict['size_class_label'].cpu().numpy()
    # all_gt_num_bbox                 = data_dict['num_bbox'].cpu().numpy()               #(B)
    # all_gt_box_label                = data_dict['ref_box_label'].cpu().numpy()          #(B)
    # gt_obj_id                       = data_dict['object_id'].cpu().numpy()              #(B,128)
    # loss_contrastive = []
    # label_done = []

    # flag_add_threshold = num_sample_contra if num_sample_contra!=None else 1
    # # Loop1: Each batch. (B)
    # for indx in range(len(all_gt_sem_labels)):
    #     current_label = gt_size_class[indx]
    #     label_done.append(current_label)
        
    #     # Initial output, the first item is the postive sample
    #     out       = np.array(pred_ref[indx][ref_iou_idx[indx]])
    #     out_label = np.array(1.0)
    #     flag_add  = 0

    #     # Loop2: Compare current batch to all batches (B)
    #     for indx2 in range(len(all_gt_sem_labels)):
    #         if flag_add >= flag_add_threshold: break
    #         item_list = all_gt_sem_labels[indx2]  # Items in the compare batch

    #         # Loop3: Each object in the same scene (128)
    #         for indx_test in range(len(item_list)):
    #             if flag_add >= flag_add_threshold: break
    #             #Stop if exceeding number of obj
    #             if indx_test >= all_gt_num_bbox[indx2]-1 :
    #                 break
                
    #             match_label_same = all_gt_box_label[indx][indx_test] if indx == indx2 else 0  # Don't add the it self twice if in the same batch
    #             # Find the object in same scene with same label
    #             if current_label == item_list[indx_test] and match_label_same != 1:
    #                 # print('Batch indx1 :', indx, 'Batch indx2 :', indx2, " item index : ", indx_test, "out :", out.shape)
    #                 # Comparing 
    #                 # get required data 
    #                 match_center_label              = all_gt_center_label[indx2][indx_test]
    #                 match_heading_class_label       = all_gt_heading_class_label[indx2][indx_test]
    #                 match_heading_residual_label    = all_gt_heading_residual_label[indx2][indx_test]
    #                 match_size_class_label          = all_gt_size_class_label[indx2][indx_test]
    #                 match_size_residual_label       = all_gt_size_residual_label[indx2][indx_test]

    #                 # convert gt bbox parameters to bbox corners
    #                 gt_obb        = config.param2obb(match_center_label[0:3], match_heading_class_label, match_heading_residual_label,
    #                                 match_size_class_label, match_size_residual_label)
    #                 gt_bbox_batch = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3])  # rename

    #                 # convert the bbox parameters to bbox corners
    #                 pred_obb_batch = config.param2obb_batch(pred_center[indx2, :, 0:3], pred_heading_class[indx2], pred_heading_residual[indx2],
    #                                     pred_size_class[indx2], pred_size_residual[indx2])
    #                 pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
    #                 # Find highest IOU to find box
    #                 ious                     = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch, (num_proposals, 1, 1)))
    #                 labels[i, ious.argmax()] = 1        # treat the bbox with highest iou score as the gt
    #                 out                      = np.append(out,pred_ref[indx2][ious.argmax()])
    #                 out_label                = np.append(out_label,0)
    #                 flag_add                 += 1
                    
    #     # out in a batch is a 'list' of boxex where the first one is positive sample
    #     if flag_add != 0:
    #         out       = torch.FloatTensor(out)
    #         out_label = torch.FloatTensor(out_label)
    #         test1     = out.float().clone()
    #         test2     = out_label.float().clone()
    #         # assert test1.shape == test2.shape
    #         # print("test1",test1.shape)
    #         # print("test2",test2.shape)
    #         loss_temp = criterion(test1, test2,dim_in=0)
    #         loss_contrastive.append(loss_temp)

    #     # Calculate the loss in each batch and append to result

    # # Mean result
    # loss_contrastive = torch.stack(loss_contrastive)
    # loss_contrastive = torch.mean(loss_contrastive)
    # loss_contrastive = loss_contrastive.cuda()
    # loss_contrastive.requires_grad=True



    # return loss, cluster_preds, cluster_labels
    return loss, loss_mask, loss_contr, cluster_preds, cluster_labels

def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(data_dict["lang_scores"], data_dict["object_cat"])

    return loss

def compute_match_box_loss(data_dict, config):
    # Need: predicionts, targets, indices from mathcer
    # TODO: Assert predict box exisits

    # Get index from matcher
    indices = data_dict["match_indices_list"]
    idx = _get_src_permutation_idx(indices)
    
    # Prepare data for box loss
    pred_boxes,gt_boxes = get_corner(data_dict,config,indices,idx)  # Selected pairs
    number_box = data_dict["num_bbox"].cpu().numpy()
    pred_center = data_dict['center'][idx]    

    # GT data : Center, Size, class label
    gt_center = data_dict['center_label'][:,:,0:3]
    gt_box_size = data_dict['size_residual_label']
    gt_class_label = data_dict['size_class_label']

    # GT data : Use index to select GT data 
    gt_class_label_list = []
    gt_box_size_list    = []
    gt_center_list      = []
    for i in range(gt_center.shape[0]):
        gt_class_label_list.append(gt_class_label[i,:number_box[i]])
        gt_box_size_list.append(gt_box_size[i,:number_box[i]])
        gt_center_list.append(gt_center[i,:number_box[i],:])
    gt_class_label  = torch.cat([torch.as_tensor(t[i]) for t, (_, i) in zip(gt_class_label_list, indices)], dim=0)
    gt_box_size     = torch.cat([torch.as_tensor(t[i]) for t, (_, i) in zip(gt_box_size_list, indices)], dim=0)                     # **                  
    gt_center       = torch.cat([torch.as_tensor(t[i]) for t, (_, i) in zip(gt_center_list, indices)], dim=0)

    # Predictions data: Box Size
    predicted_box_size = data_dict['size_residuals'][idx]
    predicted_box_size = torch.gather(predicted_box_size, 1, gt_class_label.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3)).squeeze(1)   # **

        
    ##### Loss Calculation #####
    # GIOU loss
    cost_giou = generalized_box_iou(pred_boxes,gt_boxes)
    cost_giou = torch.Tensor(cost_giou).cuda()
    giou_loss = 1 - torch.diag(cost_giou)
    giou_loss = torch.mean(giou_loss)

    # Box loss :center loss
    center_loss = F.mse_loss(pred_center,gt_center,reduction='none')
    center_loss = torch.mean(center_loss)
    # Box loss: size loss    
    size_reg_loss = huber_loss(predicted_box_size - gt_box_size, delta=1.0)
    size_reg_loss = torch.mean(size_reg_loss)

    box_loss = center_loss + size_reg_loss

    # box_loss.requires_grad=True
    giou_loss.requires_grad=True
    return box_loss, giou_loss

def computer_match_label_loss(data_dict,config):
    # Need predictions, targets,indices
    indices = data_dict["match_indices_list"]
    idx = _get_src_permutation_idx(indices)

    # Get Ground truth labels
    pred_logits = data_dict['sem_cls_scores'] # predict labels
    gt_logits = data_dict["sem_cls_label"]
    gt_num_bbox = data_dict["num_bbox"]

    # Get GT labels: Use index to select GT data 
    gt_logits_list = []
    for i in range(gt_logits.shape[0]):
        gt_logits_list.append(gt_logits[i,:gt_num_bbox[i]])
    gt_logits  = torch.cat([torch.as_tensor(t[i]) for t, (_, i) in zip(gt_logits_list, indices)], dim=0)

    # TODO: Now fixed type 18 as no object "None"
    # Fill will all none, except the matched pairs
    gt_classes = torch.full(pred_logits.shape[:2], 18,
                        dtype=torch.int64, device=pred_logits.device)

    # gt_classes = torch.full(pred_logits.shape[:2], 17,
    #                     dtype=torch.int64, device=pred_logits.device)   # For past model

    gt_classes[idx] = gt_logits


    empty_weight = torch.ones(18 + 1).cuda()
    # empty_weight = torch.ones(18 ).cuda()   # For past model
    empty_weight[-1] = 0.1

    ce_loss = F.cross_entropy(pred_logits.transpose(1, 2), gt_classes, empty_weight)

    class_error = 100 - accuracy(pred_logits[idx], gt_logits)[0]

    class_error_matched = 100 - accuracy(pred_logits[idx], gt_logits)[0]
    
    # test_pred = pred_logits[idx]
    # test_gt   = gt_logits
    # test_pred_class = test_pred.argmax(-1)
    # ce_loss_matched = F.cross_entropy(test_pred, test_gt)

    # numer_preded_all     = (pred_logits.argmax(-1) != pred_logits.shape[-1] -1 )
    # numer_preded_matched = (pred_logits[idx].argmax(-1) != pred_logits.shape[-1] -1 )

    numer_preded_all     = (pred_logits.argmax(-1) != pred_logits.shape[-1] -1 ).sum(1)
    numer_preded_matched = (pred_logits[idx].argmax(-1) != pred_logits.shape[-1] -1 ).sum(0)

    card_err_all     = F.l1_loss(numer_preded_all.float(),gt_num_bbox.float())            / gt_num_bbox.shape[0]
    card_err_matched = F.l1_loss(numer_preded_matched.float(),gt_num_bbox.sum(0).float()) / gt_num_bbox.shape[0]



    # _, test_all_pred = pred_logits.topk(1, 2, True, True)

    # pred_logits_select = data_dict['sem_cls_scores'][idx]
    # ce_select = F.cross_entropy(pred_logits_select,gt_logits)
    # pred_logits_rand = pred_logits.detach().clone()
    # pred_logits_rand = torch.rand(pred_logits_rand.shape[:3]).cuda()
    # ce_loss_rand = F.cross_entropy(pred_logits_rand.transpose(1, 2), gt_classes, empty_weight)

    return ce_loss , class_error , card_err_all, card_err_matched

def get_corner(data,config,indices,idx):
        # predicted box
        pred_center = data['center'].detach().cpu().numpy()
        # pred_size = data['size'].detach().cpu().numpy()
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

        pred_center             = pred_center[idx]
        # pred_size               = pred_size[idx]
        pred_heading_class      = pred_heading_class[idx]
        pred_heading_residual   = pred_heading_residual[idx]
        pred_size_class         = pred_size_class[idx]
        pred_size_residual      = pred_size_residual[idx]

        gt_center_list              = []
        gt_heading_class_list       = []
        gt_heading_residual_list    = []
        gt_size_class_list          = []
        gt_size_residual_list       = []
        for i in range(gt_center.shape[0]):
            gt_center_list                      .append(gt_center[i,:number_box[i],:])
            gt_heading_class_list               .append(gt_heading_class[i,:number_box[i]])
            gt_heading_residual_list            .append(gt_heading_residual[i,:number_box[i]])
            gt_size_class_list                  .append(gt_size_class[i,:number_box[i]])
            gt_size_residual_list               .append(gt_size_residual[i,:number_box[i],:])

        gt_center           = torch.cat([torch.as_tensor(t[i]) for t, (_, i) in zip(gt_center_list, indices)], dim=0).cpu().numpy()
        gt_heading_class    = torch.cat([torch.as_tensor(t[i]) for t, (_, i) in zip(gt_heading_class_list, indices)], dim=0).cpu().numpy()
        gt_heading_residual = torch.cat([torch.as_tensor(t[i]) for t, (_, i) in zip(gt_heading_residual_list, indices)], dim=0).cpu().numpy()
        gt_size_class       = torch.cat([torch.as_tensor(t[i]) for t, (_, i) in zip(gt_size_class_list, indices)], dim=0).cpu().numpy()
        gt_size_residual    = torch.cat([torch.as_tensor(t[i]) for t, (_, i) in zip(gt_size_residual_list, indices)], dim=0).cpu().numpy()

        pred_obb_batch = config.param2obb_batch(pred_center[:, 0:3], pred_heading_class, pred_heading_residual,
                        pred_size_class, pred_size_residual)
        pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])

        # pred_bbox_batch = get_3d_box_batch_no_heading(pred_size,pred_center)
            

        gt_obb_batch = config.param2obb_batch(gt_center[:, 0:3], gt_heading_class, gt_heading_residual,
                        gt_size_class, gt_size_residual)
        gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
        
        # pred_corner = []
        # gt_corner   = []
        # for i in range(pred_center.shape[0]):
        #     pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
        #                     pred_size_class[i], pred_size_residual[i])
        #     pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
        #     pred_corner.append(pred_bbox_batch)

        #     gt_obb_batch = config.param2obb_batch(gt_center[i, :number_box[i], 0:3], gt_heading_class[i,:number_box[i]], gt_heading_residual[i,:number_box[i]],
        #                     gt_size_class[i,:number_box[i]], gt_size_residual[i,:number_box[i],:])
        #     gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
        #     gt_corner.append(gt_bbox_batch)

        # pred_corner = np.stack(pred_corner)

        # gt_corner = np.vstack(gt_corner)


        return pred_bbox_batch,gt_bbox_batch

def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

def get_loss(data_dict, config, detection=True, reference=True, use_lang_classifier=False,mask_aug=False,use_matcher=False,phase='Train'):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    use_matcher = False
    if use_matcher:
        #Loss from deter
        data_dict = matcher(data_dict)
        match_box_loss, giou_loss = compute_match_box_loss(data_dict,config)
        ce_loss , class_error, card_err_all, card_err_matched = computer_match_label_loss(data_dict,config)

        loss = 5*match_box_loss + 1* ce_loss + 2*giou_loss

        data_dict['box_loss'] = match_box_loss
        data_dict['giou_loss'] = giou_loss  # TODO: Change objectness loss name to giou loss
        data_dict['ce_loss'] = ce_loss
        data_dict['class_error'] = class_error  # No grad
        data_dict['card_err_all'] = card_err_all
        data_dict['card_err_matched'] = card_err_matched

        
    else:
        # Vote loss
        # vote_loss = compute_vote_loss(data_dict)
        # Obj loss
        # if phase !='Train': 
        data_dict = matcher(data_dict)# Don't eval during train just for speed things up

        objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
        num_proposal = objectness_label.shape[1]
        total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
        data_dict['objectness_label'] = objectness_label
        data_dict['objectness_mask'] = objectness_mask
        data_dict['object_assignment'] = object_assignment
        data_dict['pos_ratio'] = torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
        data_dict['neg_ratio'] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict['pos_ratio']

        # Box loss and sem cls loss
        center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
        box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
        
        data_dict['box_loss'] = box_loss
        data_dict['objectness_loss'] = objectness_loss
        data_dict['center_loss'] = center_loss
        data_dict['heading_cls_loss'] = heading_cls_loss
        data_dict['heading_reg_loss'] = heading_reg_loss
        data_dict['size_cls_loss'] = size_cls_loss
        data_dict['size_reg_loss'] = size_reg_loss
        data_dict['sem_cls_loss'] = sem_cls_loss
        data_dict['scan_box_loss'] = box_loss

        match_box_loss, giou_loss = box_loss*0, box_loss* 0
        ce_loss , class_error, card_err_all, card_err_matched = box_loss*0, box_loss*0, box_loss*0, box_loss*0

        loss =  0.5*data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1*data_dict['sem_cls_loss'] 
        loss *= 10 # amplify

        
        data_dict['giou_loss'] = objectness_loss  # TODO: Change objectness loss name to giou loss
        data_dict['ce_loss'] = sem_cls_loss
        data_dict['class_error'] = class_error  # No grad
        data_dict['card_err_all'] = card_err_all
        data_dict['card_err_matched'] = card_err_matched
        


    

    
    
    
    # loss = 5*data_dict['box_loss'] + 1* data_dict['ce_loss'] + 2*data_dict['giou_loss'] 
    



    # # Final loss function
    data_dict['loss'] = loss
    return loss, data_dict

    # # Vote loss
    # # vote_loss = compute_vote_loss(data_dict)

    # # Obj loss
    # objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    # num_proposal = objectness_label.shape[1]
    # total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    # data_dict['objectness_label'] = objectness_label
    # data_dict['objectness_mask'] = objectness_mask
    # data_dict['object_assignment'] = object_assignment
    # data_dict['pos_ratio'] = torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    # data_dict['neg_ratio'] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict['pos_ratio']

    # # Box loss and sem cls loss
    # center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    # box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss

    # if detection:
    #     # data_dict['vote_loss'] = vote_loss
    #     data_dict['objectness_loss'] = objectness_loss
    #     data_dict['center_loss'] = center_loss
    #     data_dict['heading_cls_loss'] = heading_cls_loss
    #     data_dict['heading_reg_loss'] = heading_reg_loss
    #     data_dict['size_cls_loss'] = size_cls_loss
    #     data_dict['size_reg_loss'] = size_reg_loss
    #     data_dict['sem_cls_loss'] = sem_cls_loss
    #     data_dict['box_loss'] = box_loss
    # else:
    #     data_dict['vote_loss'] = torch.zeros(1)[0].cuda()
    #     data_dict['objectness_loss'] = torch.zeros(1)[0].cuda()
    #     data_dict['center_loss'] = torch.zeros(1)[0].cuda()
    #     data_dict['heading_cls_loss'] = torch.zeros(1)[0].cuda()
    #     data_dict['heading_reg_loss'] = torch.zeros(1)[0].cuda()
    #     data_dict['size_cls_loss'] = torch.zeros(1)[0].cuda()
    #     data_dict['size_reg_loss'] = torch.zeros(1)[0].cuda()
    #     data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
    #     data_dict['box_loss'] = torch.zeros(1)[0].cuda()

    # if reference:
    #     # Reference loss
    #     ref_loss, mask_loss, contr_loss, _, cluster_labels = compute_reference_loss(data_dict, config,mask_aug)
    #     # ref_loss = torch.zeros(1)[0].cuda()
    #     # contr_loss = torch.zeros(1)[0].cuda()
    #     # cluster_labels = objectness_label.new_zeros(objectness_label.shape).cuda()
        
    #     data_dict["cluster_labels"] = cluster_labels
    #     data_dict["ref_loss"] = ref_loss
    #     data_dict["contr_loss"] = contr_loss
    #     data_dict["mask_loss"] = mask_loss
    # else:
    #     # # Reference loss
    #     # ref_loss, contr_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
    #     # data_dict["cluster_labels"] = cluster_labels
    #     data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda()
    #     data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda()

    #     # store
    #     data_dict["ref_loss"] = torch.zeros(1)[0].cuda()
    #     # data_dict["contr_loss"] = torch.zeros(1)[0].cuda()

    # if reference and use_lang_classifier:
    #     data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    # else:
    #     data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    # Final loss function
    # loss = data_dict['vote_loss'] + 0.5*data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1*data_dict['sem_cls_loss'] \
    #      + 0.1*data_dict["lang_loss"] \
    #     + 1*data_dict["contr_loss"] \
    #     + 0.1*data_dict["mask_loss"]

        # + 0.1*data_dict["ref_loss"]
    
    

    # return loss, data_dict
