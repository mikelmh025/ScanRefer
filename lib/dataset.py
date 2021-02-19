'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import h5py
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt  
import multiprocessing as mp
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz, choose_label_pc
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis
import random


# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")

class ScannetReferenceDataset(Dataset):
       
    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False,
        cp_aug=False):

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.cp_aug = cp_aug

        # load data
        self._load_data()
        self.multiview_data = {}
       
    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        lang_len = len(self.scanrefer[idx]["token"])
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN else CONF.TRAIN.MAX_DES_LEN

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        point_cloud,pcl_color = self.process_pc(mesh_vertices,scene_id)

        # Add negative samples
        num_obj_add =  self.cp_aug
        neg_box = np.zeros((num_obj_add, 6))
        if self.cp_aug!=0 and self.split != 'test':
    
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes = np.zeros((MAX_NUM_OBJ,))
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            size_classes[0:num_bbox] = class_ind
            for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
                if gt_id == object_id:
                    ref_sem_label = instance_bboxes[i,-2]

            # for i in range(num_obj_add):
            list_obj = self.obj_lib[ref_sem_label]
            if len(list_obj) != 0: 
                selected_neg_obj = random.sample(list_obj,len(list_obj)) if len(list_obj) <= num_obj_add else random.sample(list_obj,num_obj_add)

                for (index, item) in enumerate(selected_neg_obj):
                    other_point_cloud,other_pcl_color = self.process_pc(item["mesh_vertices"],item["scene_id"])
                    point_cloud     = np.concatenate((point_cloud,other_point_cloud),axis=0) 
                    pcl_color       = np.concatenate((pcl_color,other_pcl_color),axis=0) 
                    semantic_labels = np.concatenate((semantic_labels,item["semantic_labels"]),axis=0)
                    instance_bboxes = np.concatenate((instance_bboxes,np.atleast_2d(item["instance_bboxes"])),axis=0)
                    neg_box[index,:] = item["instance_bboxes"][0:6]

                    obj_instance_labels = np.empty(item["instance_labels"].shape)
                    obj_instance_labels.fill(np.max(instance_labels) +1)

                    instance_labels = np.concatenate((instance_labels,obj_instance_labels),axis=0)
                    
        
        # if self.cp_aug and self.split != 'test':
        #     # Choose examples from other scene
        #     num_obj_add =  32 - instance_bboxes.shape[0]
        #     for i in range(num_obj_add):
        #         idx_other = random.randint(0,len(self.scanrefer)-1)
        #         while idx_other== idx:
        #             idx_other = random.randint(0,len(self.scanrefer)-1)
        #         try:
        #             other_scene_id = self.scanrefer[idx_other]["scene_id"]
        #             other_object_id = int(self.scanrefer[idx_other]["object_id"])
        #             other_object_name = " ".join(self.scanrefer[idx_other]["object_name"].split("_"))
        #             other_ann_id = self.scanrefer[idx_other]["ann_id"]
        #         except IndexError:
        #             print("Index Error: Selecting an index out of range")

        #         # get pc
        #         other_mesh_vertices = self.scene_data[other_scene_id]["mesh_vertices"]
        #         other_instance_labels = self.scene_data[other_scene_id]["instance_labels"]
        #         other_semantic_labels = self.scene_data[other_scene_id]["semantic_labels"]
        #         other_instance_bboxes = self.scene_data[other_scene_id]["instance_bboxes"]
        #         other_point_cloud,other_pcl_color = self.process_pc(other_mesh_vertices,scene_id)

        #         jitter_idx = 1#random.random()*0.45 +0.8 # Standard scale jittering

        #         # Random pick object and append to the current scene
        #         target_obj_label = random.randint(0,np.max(other_instance_labels)) # Find object based on id
        #         test_instance_labels, test_choices, flag_exceed = choose_label_pc(other_instance_labels, target_obj_label, 200, return_choices=True)
        #         test_instance_labels = np.empty(test_instance_labels.shape)
                
        #         other_point_cloud_jitter = other_point_cloud[test_choices].copy()
        #         # other_point_cloud_jitter[:,0:3] *= jitter_idx
        #         point_cloud     = np.concatenate((point_cloud,other_point_cloud_jitter),axis=0) 
        #         semantic_labels = np.concatenate((semantic_labels,other_semantic_labels[test_choices]),axis=0)
        #         pcl_color       = np.concatenate((pcl_color,other_pcl_color[test_choices]),axis=0)

        #         # Find the right box in other scene
        #         flag_add_instance = 0
        #         for i, gt_id in enumerate(other_instance_bboxes[:other_instance_bboxes.shape[0],-1]):
        #             if gt_id == other_object_id:
        #                 select = other_instance_bboxes[i].copy()
        #                 select[-1] = np.max(instance_bboxes) +1
        #                 test_instance_labels.fill(np.max(instance_bboxes) +1)
        #                 instance_bboxes = np.concatenate((instance_bboxes,np.atleast_2d(select)),axis=0)
        #                 instance_labels = np.concatenate((instance_labels,test_instance_labels),axis=0)
        #                 flag_add_instance = 1
        #                 break
        #         if flag_add_instance == 0:
        #             print("Warning: Did not add a box from another scene, something wrong")

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        
        
        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        ref_box_label = np.zeros(MAX_NUM_OBJ) # bbox label for reference target
        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target

        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            if self.augment and not self.debug:
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]
                    target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                    
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                    target_bboxes[:,1] = -1 * target_bboxes[:,1]                                

                # Rotation along X-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

                # Rotation along Y-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

                # Translation
                point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label. 
            for i_instance in np.unique(instance_labels):            
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label            
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0))
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
            
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]

            # construct the reference target label for each bbox
            ref_box_label = np.zeros(MAX_NUM_OBJ)
            for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
                if gt_id == object_id:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical 
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["lang_feat"] = lang_feat.astype(np.float32) # language feature vectors
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        # data_dict["ref_box_label"] = ref_box_label.astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_box_label"] = ref_box_label.astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)

        data_dict["neg_boxes"] = neg_box.astype(np.float32) # The boxes for the negative samples

        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(np.int64)
        # data_dict["pcl_color"] = pcl_color
        data_dict["load_time"] = time.time() - start

        # data_dict["test_point_clouds"] = test_point_cloud.astype(np.float32)
        # data_dict["test_pcl_color"] = test_pcl_color

        return data_dict
    
    def process_pc(self,mesh_vertices,scene_id):
        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")
 
            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        return point_cloud,pcl_color


    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        lang = {}
        test = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}
                test[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                test[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                # test[scene_id][object_id][ann_id] = {}

            # tokenize the description
            tokens = data["token"]
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN, 300))
            for token_id in range(CONF.TRAIN.MAX_DES_LEN):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        embeddings[token_id] = glove[token]
                    else:
                        embeddings[token_id] = glove["unk"]

            # store
            lang[scene_id][object_id][ann_id] = embeddings
            # getting data for Analysis the language instances with other objects from same category in the same scene
            # And second test as well
            # test[scene_id][object_id] = [data['object_id'],data['object_name']]

        # out_stat = {}
        # for 

        # getting data for Analysis the language instances with other objects from same category in the same scene
        # for scene in test:
        #     for obj in test[scene]:
        #         test[scene][obj].append(0)
        #         for obj_compare in test[scene]:
        #             if test[scene][obj][0] != test[scene][obj_compare][0] and test[scene][obj][1] == test[scene][obj_compare][1]:
        #                 test[scene][obj][2] += 1
        #         if str(test[scene][obj][2]) not in out_stat:
        #             out_stat[str(test[scene][obj][2])] = {}
        #             out_stat[str(test[scene][obj][2])] = 1
        #         else:
        #             out_stat[str(test[scene][obj][2])] += 1

        # # creating the dataset 
        # data = out_stat
        # keys = list(data.keys()) 
        # keys = [int(string) for string in keys]
        # values = list(data.values()) 
        
        # fig = plt.figure(figsize = (10, 5)) 
        
        # # creating the bar plot 
        # plt.bar(keys, values, color ='maroon',  
        #         width = 0.4) 
        
        # plt.xlabel("Number of other object from same class") 
        # plt.ylabel("Number of annotation") 
        # plt.title("Analysis the language instances with other objects from same category") 
        # plt.show()

        # plt.pie(values, labels = keys)
        # plt.show()

        return lang


    def _load_data(self):
        print("loading data...")
        # load language features
        self.lang = self._tranform_des()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))
        
        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        self.obj_lib = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name
            self.obj_lib[nyu40_name] = []

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

            instance_labels = self.scene_data[scene_id]["instance_labels"]
            for label in range(np.min(instance_labels), np.max(instance_labels)):
                selected_instance_labels, choices, _ = choose_label_pc(instance_labels,label,200,return_choices=True)
                selected_semantic_labels = self.scene_data[scene_id]["semantic_labels"][choices]
                selected_mesh_vertices = self.scene_data[scene_id]["mesh_vertices"][choices]
                selected_instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

                obj_data = {}
                obj_data["scene_id"] = scene_id
                obj_data["mesh_vertices"] = selected_mesh_vertices
                obj_data["instance_labels"] = selected_instance_labels
                obj_data["semantic_labels"] = selected_semantic_labels
                # TODO: Make this work, check sematic 
                # obj_data["instance_bboxes"] = selected_instance_bboxes
                add_flag = 0
                for i, gt_id in enumerate(selected_instance_bboxes[:selected_instance_bboxes.shape[0],-1]):
                    if gt_id == np.min(selected_instance_labels):
                        obj_data["instance_bboxes"] = selected_instance_bboxes[i].copy()
                        add_flag = 1
                        break
                        # instance_bboxes = np.concatenate((instance_bboxes,np.atleast_2d(select)),axis=0)

                # If has box, Append object to library
                if add_flag == 1 :
                    if np.max(selected_semantic_labels) != 0:
                        self.obj_lib[np.max(selected_semantic_labels)].append(obj_data) 

        
                            
                

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

        # # Loop through each scene
        # for scene_id in self.scene_list:
        #     # loop through each objects
        #     instance_labels = self.scene_data[scene_id]["instance_labels"]
        #     for label in range(np.min(instance_labels), np.max(instance_labels)):
        #         selected_instance_labels, choices, _ = choose_label_pc(instance_labels,label,200,return_choices=True)
        #         selected_semantic_labels = self.scene_data[scene_id]["semantic_labels"][choices]
        #         selected_mesh_vertices = self.scene_data[scene_id]["mesh_vertices"][choices]
        #         selected_instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        #         obj_data = {}
        #         obj_data["mesh_vertices"] = selected_mesh_vertices
        #         obj_data["instance_labels"] = selected_instance_labels
        #         obj_data["semantic_labels"] = selected_semantic_labels
        #         # TODO: Make this work
        #         obj_data["instance_bboxes"] = selected_instance_bboxes
        #         obj_data["scene_id"] = scene_id
                
        #         #Append object to library
        #         if np.max(selected_semantic_labels) != 0:
        #             self.obj_lib[np.max(selected_semantic_labels)].append(obj_data) 
                
        #         # selected_instance_bboxes
            
            
        #     self.scene_data[scene_id]["instance_bboxes"]

            

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
