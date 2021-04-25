import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.encoder_module import EncoderModule
from models.decoder_module import DecoderModule


from models.proposal_module import ProposalModule
from models.lang_module import LangModule
from models.match_module import MatchModule

from models.transformer_module import TransformerModule
from models.combine_module import CombineModule
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from transformers import BertModel

# from models.matcher import build_matcher


class RefNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
    use_lang_classifier=True, use_bidir=False, no_reference=False,attn=False,mask_aug=False,
    emb_size=300, hidden_size=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir      
        self.no_reference = no_reference
        self.attn = attn
        self.mask_aug=mask_aug

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim,attn=self.attn)
        self.encoder = EncoderModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)
        self.decoder = DecoderModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)
        self.combiner = CombineModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

        # Hough voting
        # self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size,attn,mask_aug)

        #     # --------- PROPOSAL MATCHING ---------
        #     # Match the generated proposals and select the most confident ones
        #     self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size,mask_aug=self.mask_aug)

        # # self.model_bert = BertModel.from_pretrained('bert-base-cased')
        self.Linear_bert_out1 = nn.Conv1d(768, 512, kernel_size=1, bias=False)
        self.Linear_bert_out2 = nn.Conv1d(512, 512, kernel_size=1, bias=False)
        self.Linear_bert_out3 = nn.Conv1d(512, 256, kernel_size=1, bias=False)

        self.Linear_bert_out_pool1 = nn.Conv1d(768, 512, kernel_size=1, bias=False)
        self.Linear_bert_out_pool2 = nn.Conv1d(512, 512, kernel_size=1, bias=False)
        self.Linear_bert_out_pool3 = nn.Conv1d(512, 256, kernel_size=1, bias=False)

        # self.TransformerModule = TransformerModule()
        

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
                        #  Bert 
            bert_in = {}
                        # bert_in['input_ids'] = data_dict["bertTo_input"].squeeze(1)
                        # bert_in["token_type_ids"] = data_dict["bertTo_type"].squeeze(1)
                        # bert_in["attention_mask"] = data_dict["bertTo_mask"].squeeze(1)
                        # with torch.no_grad():
                        #     bert_out = self.model_bert(**bert_in)

            # data_dict["bert_hidden"] = data_dict["bert_hidden"].squeeze(1)
            # data_dict["bert_poolar"] = data_dict["bert_poolar"].squeeze(1)
            
            # bert_out_hidden = self.Linear_bert_out1(data_dict["bert_hidden"].transpose(1,2))
            # bert_out_hidden = self.Linear_bert_out2(bert_out_hidden)
            # bert_out_hidden = self.Linear_bert_out3(bert_out_hidden).transpose(2,1)
            # data_dict["bert_out_hidden"] = bert_out_hidden

            
            # bert_out_pool = self.Linear_bert_out_pool1(data_dict["bert_poolar"].unsqueeze(2))
            # bert_out_pool = self.Linear_bert_out_pool2(bert_out_pool)
            # bert_out_pool = self.Linear_bert_out_pool3(bert_out_pool).squeeze(2)
            # data_dict["bert_out_pool"] = bert_out_pool

                
            data_dict = self.lang(data_dict)

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        
        data_dict = self.backbone_net(data_dict)
        


                
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        # data_dict["seed_inds"] = data_dict["fp2_inds"]
        # data_dict["seed_xyz"] = xyz
        # data_dict["seed_features"] = features

        
        
        # xyz, features = self.vgen(xyz, features)
        # features_norm = torch.norm(features, p=2, dim=1)
        # features = features.div(features_norm.unsqueeze(1))
        # data_dict["vote_xyz"] = xyz
        # data_dict["vote_features"] = features



        # --------- PROPOSAL GENERATION ---------   
        data_dict = self.proposal(xyz, features, data_dict)
        # data_dict = self.encoder(data_dict)
        # data_dict = self.decoder(data_dict)

        
        # f = data_dict['aggregated_vote_features']
        # b = data_dict["bert_out_hidden"]

        data_dict = self.combiner(data_dict)
        # data_dict = self.TransformerModule(data_dict)


        #########################3

        # if not self.no_reference:

        #     #######################################
        #     #                                     #
        #     #          PROPOSAL MATCHING          #
        #     #                                     #
        #     #######################################

        #     # --------- PROPOSAL MATCHING ---------
        #     data_dict = self.match(data_dict)

        return data_dict
