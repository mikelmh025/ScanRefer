import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
import scipy.optimize


sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from models.refnet import RefNet
from data.scannet.model_util_scannet import ScannetDatasetConfig
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

def get_dataloader(args, scanrefer, all_scene_list, split, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_color=args.use_color, 
        use_height=(not args.no_height),
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        bert_emb=False
    )
    print("evaluate on {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataset, dataloader

def get_model(args, config):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = RefNet(
        num_class=config.num_class,
        num_heading_bin=config.num_heading_bin,
        num_size_cluster=config.num_size_cluster,
        mean_size_arr=config.mean_size_arr,
        num_proposal=args.num_proposals,
        input_feature_dim=input_channels,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        attn=args.self_attn,
    ).cuda()

    devices = [int(x) for x in args.devices]
    print("devices", devices, "torch.cuda.device_count()", torch.cuda.device_count())
    model = nn.DataParallel(model, device_ids=devices)

    model_name = "model_last.pth" if args.detection else "model.pth"
    path = os.path.join(CONF.PATH.BASE, args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(args):
    if args.detection:
        scene_list = get_scannet_scene_list("val")
        scanrefer = []
        for scene_id in scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            scanrefer.append(data)
    else:
        scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
        scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
        if args.num_scenes != -1:
            scene_list = scene_list[:args.num_scenes]

        scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

    return scanrefer, scene_list

def eval_ref(args):
    print("evaluate localization...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)

    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    } if not args.no_nms else None

    # random seeds
    seeds = [args.seed] + [2 * i for i in range(args.repeat - 1)]

    # evaluate
    print("evaluating...")
    score_path = os.path.join(CONF.PATH.BASE, args.folder, "scores.p")
    pred_path = os.path.join(CONF.PATH.BASE, args.folder, "predictions.p")
    gen_flag = (not os.path.exists(score_path)) or args.force or args.repeat > 1
    if gen_flag:
        ref_acc_all = []
        ious_all = []
        masks_all = []
        others_all = []
        lang_acc_all = []
        for seed in seeds:
            # reproducibility
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            print("generating the scores for seed {}...".format(seed))
            ref_acc = []
            ious = []
            masks = []
            others = []
            lang_acc = []
            predictions = {}
            for data in tqdm(dataloader):
                for key in data:
                    data[key] = data[key].cuda()

                # feed
                data = model(data)
                _, data = get_loss(
                    data_dict=data, 
                    config=DC, 
                    detection=True,
                    reference=True, 
                    use_lang_classifier=not args.no_lang_cls
                )
                data = get_eval(
                    data_dict=data, 
                    config=DC,
                    reference=True, 
                    use_lang_classifier=not args.no_lang_cls,
                    use_oracle=args.use_oracle,
                    use_cat_rand=args.use_cat_rand,
                    use_best=args.use_best,
                    post_processing=POST_DICT
                )

                ref_acc += data["ref_acc"]
                ious += data["ref_iou"]
                masks += data["ref_multiple_mask"]
                others += data["ref_others_mask"]
                lang_acc.append(data["lang_acc"].item())

                # store predictions
                ids = data["scan_idx"].detach().cpu().numpy()
                for i in range(ids.shape[0]):
                    idx = ids[i]
                    scene_id = scanrefer[idx]["scene_id"]
                    object_id = scanrefer[idx]["object_id"]
                    ann_id = scanrefer[idx]["ann_id"]

                    if scene_id not in predictions:
                        predictions[scene_id] = {}

                    if object_id not in predictions[scene_id]:
                        predictions[scene_id][object_id] = {}

                    if ann_id not in predictions[scene_id][object_id]:
                        predictions[scene_id][object_id][ann_id] = {}

                    predictions[scene_id][object_id][ann_id]["pred_bbox"] = data["pred_bboxes"][i]
                    predictions[scene_id][object_id][ann_id]["gt_bbox"] = data["gt_bboxes"][i]
                    predictions[scene_id][object_id][ann_id]["iou"] = data["ref_iou"][i]

            # save the last predictions
            with open(pred_path, "wb") as f:
                pickle.dump(predictions, f)

            # save to global
            ref_acc_all.append(ref_acc)
            ious_all.append(ious)
            masks_all.append(masks)
            others_all.append(others)
            lang_acc_all.append(lang_acc)

        # convert to numpy array
        ref_acc = np.array(ref_acc_all)
        ious = np.array(ious_all)
        masks = np.array(masks_all)
        others = np.array(others_all)
        lang_acc = np.array(lang_acc_all)

        # save the global scores
        with open(score_path, "wb") as f:
            scores = {
                "ref_acc": ref_acc_all,
                "ious": ious_all,
                "masks": masks_all,
                "others": others_all,
                "lang_acc": lang_acc_all
            }
            pickle.dump(scores, f)

    else:
        print("loading the scores...")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)

            # unpack
            ref_acc = np.array(scores["ref_acc"])
            ious = np.array(scores["ious"])
            masks = np.array(scores["masks"])
            others = np.array(scores["others"])
            lang_acc = np.array(scores["lang_acc"])

    multiple_dict = {
        "unique": 0,
        "multiple": 1
    }
    others_dict = {
        "not_in_others": 0,
        "in_others": 1
    }

    # evaluation stats
    stats = {k: np.sum(masks[0] == v) for k, v in multiple_dict.items()}
    stats["overall"] = masks[0].shape[0]
    stats = {}
    for k, v in multiple_dict.items():
        stats[k] = {}
        for k_o, v_o in others_dict.items():
            stats[k][k_o] = np.sum(np.logical_and(masks[0] == v, others[0] == v_o))

        stats[k]["overall"] = np.sum(masks[0] == v)

    stats["overall"] = {}
    for k_o, v_o in others_dict.items():
        stats["overall"][k_o] = np.sum(others[0] == v_o)
    
    stats["overall"]["overall"] = masks[0].shape[0]

    # aggregate scores
    scores = {}
    for k, v in multiple_dict.items():
        for k_o in others_dict.keys():
            ref_accs, acc_025ious, acc_05ious = [], [], []
            for i in range(masks.shape[0]):
                running_ref_acc = np.mean(ref_acc[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])]) \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_025iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.25)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.5)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0

                # store
                ref_accs.append(running_ref_acc)
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)

            if k not in scores:
                scores[k] = {k_o: {} for k_o in others_dict.keys()}

            scores[k][k_o]["ref_acc"] = np.mean(ref_accs)
            scores[k][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
            scores[k][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][masks[i] == multiple_dict[k]]) if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.25)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.5)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        scores[k]["overall"] = {}
        scores[k]["overall"]["ref_acc"] = np.mean(ref_accs)
        scores[k]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
        scores[k]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    scores["overall"] = {}
    for k_o in others_dict.keys():
        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][others[i] == others_dict[k_o]]) if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.25)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.5)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        # aggregate
        scores["overall"][k_o] = {}
        scores["overall"][k_o]["ref_acc"] = np.mean(ref_accs)
        scores["overall"][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
        scores["overall"][k_o]["acc@0.5iou"] = np.mean(acc_05ious)
   
    ref_accs, acc_025ious, acc_05ious = [], [], []
    for i in range(masks.shape[0]):
        running_ref_acc = np.mean(ref_acc[i])
        running_acc_025iou = ious[i][ious[i] >= 0.25].shape[0] / ious[i].shape[0]
        running_acc_05iou = ious[i][ious[i] >= 0.5].shape[0] / ious[i].shape[0]

        # store
        ref_accs.append(running_ref_acc)
        acc_025ious.append(running_acc_025iou)
        acc_05ious.append(running_acc_05iou)

    # aggregate
    scores["overall"]["overall"] = {}
    scores["overall"]["overall"]["ref_acc"] = np.mean(ref_accs)
    scores["overall"]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
    scores["overall"]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    # report
    print("\nstats:")
    for k_s in stats.keys():
        for k_o in stats[k_s].keys():
            print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

    for k_s in scores.keys():
        print("\n{}:".format(k_s))
        for k_m in scores[k_s].keys():
            for metric in scores[k_s][k_m].keys():
                print("{} | {} | {}: {}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))

    print("\nlanguage classification accuracy: {}".format(np.mean(lang_acc)))

def eval_matcher(args):
    print("evaluate detection...")
    # constant
    DC = ScannetDatasetConfig()
    
    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)

    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    sem_acc = []
    loss = []
    loss_obj = []
    loss_sem = []
    loss_box = []
    loss_giou = []
    loss_ce = []
    loss_class = []

    iou25_list = []
    iou5_list = []
    iou75_list = []
    for data in tqdm(dataloader):
        for key in data:
            data[key] = data[key].cuda()

        # feed
        with torch.no_grad():
            data = model(data)
            _, data = get_loss(
                data_dict=data, 
                config=DC, 
                detection=True,
                reference=False
            )
            

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

            indices = data["match_indices_list"]
            idx = _get_src_permutation_idx(indices)
            pred_center             = pred_center[idx]
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

            pred_obb_batch = DC.param2obb_batch(pred_center[:, 0:3], pred_heading_class, pred_heading_residual,
                            pred_size_class, pred_size_residual)
            pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])


            gt_obb_batch = DC.param2obb_batch(gt_center[:, 0:3], gt_heading_class, gt_heading_residual,
                            gt_size_class, gt_size_residual)
            gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
            
            # NUM_GT x NUM_PRED
            n_true = number_box[i]
            n_pred = pred_bbox_batch.shape[0]
            

            iou25, iou5, iou75  = match_bboxes(gt_bbox_batch,pred_bbox_batch,n_true,n_pred,IOU_THRESH=0.25)
            iou25_list.append(iou25)
            iou5_list.append(iou5)
            iou75_list.append(iou75)

            loss.append(data["loss"])
            loss_box.append(data["box_loss"])
            loss_giou.append(data["giou_loss"])
            loss_ce.append(data["ce_loss"])
            loss_class.append(data["class_loss"])
            

    loss = sum(loss)/len(loss)
    loss_box = sum(loss_box)/len(loss_box)
    loss_giou = sum(loss_giou)/len(loss_giou)
    loss_ce = sum(loss_ce)/len(loss_ce)
    loss_class = sum(loss_class)/len(loss_class)

    iou25_list = sum(iou25_list)/len(iou25_list)
    iou5_list = sum(iou5_list)/len(iou5_list)
    iou75_list = sum(iou75_list)/len(iou75_list)

    print("loss: ", loss)
    # print("loss_obj: ", loss_obj)
    # print("loss_sem: ", loss_sem)
    print("loss_box: ", loss_box)
    print("loss_giou: ", loss_giou)
    print("loss_ce: ", loss_ce)
    print("loss_class: ", loss_class)

    print("iou25_list: ",iou25_list)
    print("iou5_list: ",iou5_list)
    print("iou75_list: ",iou75_list)
 
def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
        
def eval_det(args):
    print("evaluate detection...")
    # constant
    DC = ScannetDatasetConfig()
    
    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)

    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    sem_acc = []
    loss = []
    loss_obj = []
    loss_sem = []
    loss_box = []
    loss_giou = []
    loss_ce = []
    loss_class = []

    iou25_list = []
    iou5_list = []
    iou75_list = []
    for data in tqdm(dataloader):
        for key in data:
            data[key] = data[key].cuda()

        # feed
        with torch.no_grad():
            data = model(data)
            _, data = get_loss(
                data_dict=data, 
                config=DC, 
                detection=True,
                reference=False
            )
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


            for i in range(pred_center.shape[0]):
                # convert the bbox parameters to bbox corners
                pred_obb_batch = DC.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                            pred_size_class[i], pred_size_residual[i])
                pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])


                gt_obb_batch = DC.param2obb_batch(gt_center[i, :number_box[i], 0:3], gt_heading_class[i,:number_box[i]], gt_heading_residual[i,:number_box[i]],
                                gt_size_class[i,:number_box[i]], gt_size_residual[i,:number_box[i],:])
                gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])


                # NUM_GT x NUM_PRED
                n_true = number_box[i]
                n_pred = pred_bbox_batch.shape[0]
                

                iou25, iou5, iou75  = match_bboxes(gt_bbox_batch,pred_bbox_batch,n_true,n_pred,IOU_THRESH=0.25)
                iou25_list.append(iou25)
                iou5_list.append(iou5)
                iou75_list.append(iou75)
                # ratio = ious.shape[0]/
                # print("batch: ",i, " idx_gt, idx_pred, ious, label",idx_gt, idx_pred, ious, label)




            loss.append(data["loss"])
            loss_box.append(data["box_loss"])
            loss_giou.append(data["giou_loss"])
            loss_ce.append(data["ce_loss"])
            loss_class.append(data["class_loss"])
            
            # loss_obj.append(data["objectness_loss"])
            # loss_sem.append(data["sem_cls_loss"])
    
    loss = sum(loss)/len(loss)
    # loss_obj = sum(loss_obj)/len(loss_obj)
    # loss_sem = sum(loss_sem)/len(loss_sem)
    loss_box = sum(loss_box)/len(loss_box)
    loss_giou = sum(loss_giou)/len(loss_giou)
    loss_ce = sum(loss_ce)/len(loss_ce)
    loss_class = sum(loss_class)/len(loss_class)

    iou25_list = sum(iou25_list)/len(iou25_list)
    iou5_list = sum(iou5_list)/len(iou5_list)
    iou75_list = sum(iou75_list)/len(iou75_list)

    print("loss: ", loss)
    # print("loss_obj: ", loss_obj)
    # print("loss_sem: ", loss_sem)
    print("loss_box: ", loss_box)
    print("loss_giou: ", loss_giou)
    print("loss_ce: ", loss_ce)
    print("loss_class: ", loss_class)

    print("iou25_list: ",iou25_list)
    print("iou5_list: ",iou5_list)
    print("iou75_list: ",iou75_list)
    

    #         data = get_eval_cu(
    #             data_dict=data, 
    #             config=DC, 
    #             reference=False,
    #             post_processing=POST_DICT
    #         )

    #     sem_acc.append(data["sem_acc"].item())

    #     batch_pred_map_cls = parse_predictions(data, POST_DICT) 
    #     batch_gt_map_cls = parse_groundtruths(data, POST_DICT) 
    #     for ap_calculator in AP_CALCULATOR_LIST:
    #         ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # # aggregate object detection results and report
    # print("\nobject detection sem_acc: {}".format(np.mean(sem_acc)))
    # for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
    #     print()
    #     print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
    #     metrics_dict = ap_calculator.compute_metrics()
    #     for key in metrics_dict:
    #         print("eval %s: %f"%(key, metrics_dict[key]))

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
    sel_valid = (ious_actual > 0.75)
    iou75 = ious_actual[sel_valid].shape[0]/n_true
    return iou25, iou5, iou75
    # label = sel_valid.astype(int)
    # return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 


def get_eval_cu(data_dict, config, reference, use_lang_classifier=False, use_oracle=False, use_cat_rand=False, use_best=False, post_processing=None):
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


    pred_ref = torch.argmax(data_dict['cluster_ref'] * pred_masks, 1) # (B,)
    # store the calibrated predictions and masks
    data_dict['cluster_ref'] = data_dict['cluster_ref'] * pred_masks


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



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0,1,2,3")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--force", action="store_true", help="enforce the generation of results")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times for evaluation")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_oracle", action="store_true", help="Use ground truth bounding boxes.")
    parser.add_argument("--use_cat_rand", action="store_true", help="Use randomly selected bounding boxes from correct categories as outputs.")
    parser.add_argument("--use_best", action="store_true", help="Use best bounding boxes as outputs.")
    parser.add_argument("--reference", action="store_true", help="evaluate the reference localization results")
    parser.add_argument("--detection", action="store_true", help="evaluate the object detection results")
    parser.add_argument("--matchdet", action="store_true", help="evaluate the object detection with matcher results")
    parser.add_argument("--devices", nargs='+', type=str, default=['0', '1', '2', '3'], help="devices to use")

    parser.add_argument("--cp_aug", type=int, default=0, help="number of negative sample augmentation")
    parser.add_argument("--self_attn", action="store_true", help="Use self attn in pointNet")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # evaluate
    if args.matchdet: eval_matcher(args)
    elif args.reference: eval_ref(args)
    elif args.detection: eval_det(args)
    

