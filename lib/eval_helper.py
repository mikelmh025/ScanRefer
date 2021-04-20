# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, get_3d_box_batch_no_heading
import scipy.optimize

def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one reference prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = box3d_iou(pred_bbox, gt_bbox)

    return iou

def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx/2, sx/2, -sx/2, -sx/2, sx/2, sx/2, -sx/2, -sx/2]
    y_corners = [sy/2, -sy/2, -sy/2, sy/2, sy/2, -sy/2, -sy/2, sy/2]
    z_corners = [sz/2, sz/2, sz/2, sz/2, -sz/2, -sz/2, -sz/2, -sz/2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def get_eval_cu(data,config,phase):

    # if phase == "train":
    #     data["eval_iou25"] = 0
    #     data["eval_iou5"] = 0
    #     return data

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

    # # ground truth bbox
    # gt_center = data['center_label'].cpu().numpy() # (B,128,3)
    # number_box = data["num_bbox"].cpu().numpy() #B
    # gt_heading_class = data['heading_class_label'].cpu().numpy() # B,128
    # gt_heading_residual = data['heading_residual_label'].cpu().numpy() # B,128
    # gt_size_class = data['size_class_label'].cpu().numpy() # B,128
    # gt_size_residual = data['size_residual_label'].cpu().numpy() # B,128,3

    indices = data["match_indices_list"]
    idx = _get_src_permutation_idx(indices)
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
    
    # NUM_GT x NUM_PRED
    n_true = number_box[i]
    n_pred = pred_bbox_batch.shape[0]
    

    iou25, iou5  = match_bboxes(gt_bbox_batch,pred_bbox_batch,n_true,n_pred,IOU_THRESH=0.25)
    data["eval_iou25"] = iou25
    data["eval_iou5"] = iou5

    return data

    # iou25_list.append(iou25)
    # iou5_list.append(iou5)

    # loss.append(data["loss"])
    # loss_box.append(data["box_loss"])
    # loss_giou.append(data["giou_loss"])
    # loss_ce.append(data["ce_loss"])
    # loss_class.append(data["class_error"])
    # error_card_all.append(data["card_err_all"])
    # error_card_matched.append(data["card_err_matched"])

    # iou25_list = []
    # iou5_list = []

    # for i in range(pred_center.shape[0]):
    #     # convert the bbox parameters to bbox corners
    #     pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
    #                 pred_size_class[i], pred_size_residual[i])
    #     pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])


    #     gt_obb_batch = config.param2obb_batch(gt_center[i, :number_box[i], 0:3], gt_heading_class[i,:number_box[i]], gt_heading_residual[i,:number_box[i]],
    #                     gt_size_class[i,:number_box[i]], gt_size_residual[i,:number_box[i],:])
    #     gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])


    #     # NUM_GT x NUM_PRED
    #     n_true = number_box[i]
    #     n_pred = pred_bbox_batch.shape[0]
        

    #     iou25, iou5  = match_bboxes(gt_bbox_batch,pred_bbox_batch,n_true,n_pred,IOU_THRESH=0.25)
    #     iou25_list.append(iou25)
    #     iou5_list.append(iou5)

    # #     loss.append(data["loss"])
    # #     loss_obj.append(data["objectness_loss"])
    # #     loss_box.append(data["box_loss"])
    # #     loss_sem.append(data["sem_cls_loss"])
    # # loss = sum(loss)/len(loss)
    # # loss_obj = sum(loss_obj)/len(loss_obj)
    # # loss_sem = sum(loss_sem)/len(loss_sem)
    # # loss_box = sum(loss_box)/len(loss_box)
    # iou25_list = sum(iou25_list)/len(iou25_list)
    # iou5_list = sum(iou5_list)/len(iou5_list)
    # data["eval_iou25"] = iou25
    # data["eval_iou5"] = iou5

    # return data

def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx       

def match_bboxes(bbox_gt, bbox_pred, n_true,n_pred, IOU_THRESH=0.5):
    '''
    modified version of https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
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
    # n_true = bbox_gt.shape[0]
    # n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for gt_item in range(n_true):
        for pre_item in range(n_pred):
            iou_matrix[gt_item, pre_item] = box3d_iou(bbox_gt[gt_item],bbox_pred[pre_item])

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


    sel_valid = (ious_actual > 0.25)
    iou25 = ious_actual[sel_valid].shape[0]/n_true
    sel_valid = (ious_actual > 0.5)
    iou5 = ious_actual[sel_valid].shape[0]/n_true
    # sel_valid = (ious_actual > 0.75)
    # iou75 = ious_actual[sel_valid].shape[0]/n_true
    return iou25, iou5
    # label = sel_valid.astype(int)
    # return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 


def get_eval(data_dict, config, reference, use_lang_classifier=False, use_oracle=False, use_cat_rand=False, use_best=False, post_processing=None):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    batch_size, num_words, _ = data_dict["lang_feat"].shape


    objectness_preds_batch = torch.argmax(data_dict['objectness_scores'], 2).long()
    objectness_labels_batch = data_dict['objectness_label'].long()

    if post_processing:
        _ = parse_predictions(data_dict, post_processing)
        nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()

        # construct valid mask
        pred_masks = (nms_masks * objectness_preds_batch == 1).float()
        label_masks = (objectness_labels_batch == 1).float()
    else:
        # construct valid mask
        pred_masks = (objectness_preds_batch == 1).float()
        label_masks = (objectness_labels_batch == 1).float()

    cluster_preds = torch.argmax(data_dict["cluster_ref"] * pred_masks, 1).long().unsqueeze(1).repeat(1, pred_masks.shape[1])
    preds = torch.zeros(pred_masks.shape).cuda()
    preds = preds.scatter_(1, cluster_preds, 1)
    cluster_preds = preds
    cluster_labels = data_dict["cluster_labels"].float()
    cluster_labels *= label_masks
    
    # compute classification scores
    corrects = torch.sum((cluster_preds == 1) * (cluster_labels == 1), dim=1).float()
    labels = torch.ones(corrects.shape[0]).cuda()
    ref_acc = corrects / (labels + 1e-8)
    
    # store
    data_dict["ref_acc"] = ref_acc.cpu().numpy().tolist()

    # compute localization metrics
    if use_best:
        pred_ref = torch.argmax(data_dict["cluster_labels"], 1) # (B,)
        # store the calibrated predictions and masks
        data_dict['cluster_ref'] = data_dict["cluster_labels"]
    if use_cat_rand:
        cluster_preds = torch.zeros(cluster_labels.shape).cuda()
        for i in range(cluster_preds.shape[0]):
            num_bbox = data_dict["num_bbox"][i]
            sem_cls_label = data_dict["sem_cls_label"][i]
            # sem_cls_label = torch.argmax(end_points["sem_cls_scores"], 2)[i]
            sem_cls_label[num_bbox:] -= 1
            candidate_masks = torch.gather(sem_cls_label == data_dict["object_cat"][i], 0, data_dict["object_assignment"][i])
            candidates = torch.arange(cluster_labels.shape[1])[candidate_masks]
            try:
                chosen_idx = torch.randperm(candidates.shape[0])[0]
                chosen_candidate = candidates[chosen_idx]
                cluster_preds[i, chosen_candidate] = 1
            except IndexError:
                cluster_preds[i, candidates] = 1
        
        pred_ref = torch.argmax(cluster_preds, 1) # (B,)
        # store the calibrated predictions and masks
        data_dict['cluster_ref'] = cluster_preds
    else:
        pred_ref = torch.argmax(data_dict['cluster_ref'] * pred_masks, 1) # (B,)
        # store the calibrated predictions and masks
        data_dict['cluster_ref'] = data_dict['cluster_ref'] * pred_masks

    if use_oracle:
        pred_center = data_dict['center_label'] # (B,MAX_NUM_OBJ,3)
        pred_heading_class = data_dict['heading_class_label'] # B,K2
        pred_heading_residual = data_dict['heading_residual_label'] # B,K2
        pred_size_class = data_dict['size_class_label'] # B,K2
        pred_size_residual = data_dict['size_residual_label'] # B,K2,3

        # assign
        pred_center = torch.gather(pred_center, 1, data_dict["object_assignment"].unsqueeze(2).repeat(1, 1, 3))
        pred_heading_class = torch.gather(pred_heading_class, 1, data_dict["object_assignment"])
        pred_heading_residual = torch.gather(pred_heading_residual, 1, data_dict["object_assignment"]).unsqueeze(-1)
        pred_size_class = torch.gather(pred_size_class, 1, data_dict["object_assignment"])
        pred_size_residual = torch.gather(pred_size_residual, 1, data_dict["object_assignment"].unsqueeze(2).repeat(1, 1, 3))
    else:
        pred_center = data_dict['center'] # (B,K,3)
        pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2) # B,num_proposal
        pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class
        pred_size_residual = pred_size_residual.squeeze(2) # B,num_proposal,3

    # store
    data_dict["pred_mask"] = pred_masks
    data_dict["label_mask"] = label_masks
    data_dict['pred_center'] = pred_center
    data_dict['pred_heading_class'] = pred_heading_class
    data_dict['pred_heading_residual'] = pred_heading_residual
    data_dict['pred_size_class'] = pred_size_class
    data_dict['pred_size_residual'] = pred_size_residual

    gt_ref = torch.argmax(data_dict["ref_box_label"], 1)
    gt_center = data_dict['center_label'] # (B,MAX_NUM_OBJ,3)
    gt_heading_class = data_dict['heading_class_label'] # B,K2
    gt_heading_residual = data_dict['heading_residual_label'] # B,K2
    gt_size_class = data_dict['size_class_label'] # B,K2
    gt_size_residual = data_dict['size_residual_label'] # B,K2,3

    ious = []
    multiple = []
    others = []
    pred_bboxes = []
    gt_bboxes = []
    for i in range(pred_ref.shape[0]):
        # compute the iou
        pred_ref_idx, gt_ref_idx = pred_ref[i], gt_ref[i]
        pred_obb = config.param2obb(
            pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy(), 
            pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(), 
            pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
            pred_size_class[i, pred_ref_idx].detach().cpu().numpy(), 
            pred_size_residual[i, pred_ref_idx].detach().cpu().numpy()
        )
        gt_obb = config.param2obb(
            gt_center[i, gt_ref_idx, 0:3].detach().cpu().numpy(), 
            gt_heading_class[i, gt_ref_idx].detach().cpu().numpy(), 
            gt_heading_residual[i, gt_ref_idx].detach().cpu().numpy(),
            gt_size_class[i, gt_ref_idx].detach().cpu().numpy(), 
            gt_size_residual[i, gt_ref_idx].detach().cpu().numpy()
        )
        pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
        gt_bbox = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3])
        iou = eval_ref_one_sample(pred_bbox, gt_bbox)
        ious.append(iou)

        # NOTE: get_3d_box() will return problematic bboxes
        pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
        gt_bbox = construct_bbox_corners(gt_obb[0:3], gt_obb[3:6])
        pred_bboxes.append(pred_bbox)
        gt_bboxes.append(gt_bbox)

        # construct the multiple mask
        multiple.append(data_dict["unique_multiple"][i].item())

        # construct the others mask
        flag = 1 if data_dict["object_cat"][i] == 17 else 0
        others.append(flag)

    # lang
    if reference and use_lang_classifier:
        data_dict["lang_acc"] = (torch.argmax(data_dict['lang_scores'], 1) == data_dict["object_cat"]).float().mean()
    else:
        data_dict["lang_acc"] = torch.zeros(1)[0].cuda()

    # store
    data_dict["ref_iou"] = ious
    data_dict["ref_iou_rate_0.25"] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
    data_dict["ref_iou_rate_0.5"] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]
    data_dict["ref_multiple_mask"] = multiple
    data_dict["ref_others_mask"] = others
    data_dict["pred_bboxes"] = pred_bboxes
    data_dict["gt_bboxes"] = gt_bboxes

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(data_dict['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==data_dict['objectness_label'].long()).float()*data_dict['objectness_mask'])/(torch.sum(data_dict['objectness_mask'])+1e-6)
    data_dict['obj_acc'] = obj_acc
    # detection semantic classification
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, data_dict['object_assignment']) # select (B,K) from (B,K2)
    sem_cls_pred = data_dict['sem_cls_scores'].argmax(-1) # (B,K)
    sem_match = (sem_cls_label == sem_cls_pred).float()
    data_dict["sem_acc"] = (sem_match * data_dict["pred_mask"]).sum() / data_dict["pred_mask"].sum()

    return data_dict
