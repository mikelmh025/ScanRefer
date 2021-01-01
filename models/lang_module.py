import os
import sys
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False, 
        emb_size=300, hidden_size=256):
        super().__init__() 

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        lang_size = hidden_size * 2 if self.use_bidir else hidden_size

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes),
                nn.Dropout()
            )


    def forward(self, data_dict):
        """
        encode the input descriptions
        """

        word_embs = data_dict["lang_feat"]
        print("word_embs",word_embs.shape)
        print("data_dict[lang_len]",data_dict["lang_len"].shape)
        lang_feat = pack_padded_sequence(word_embs, data_dict["lang_len"], batch_first=True, enforce_sorted=False)
    
        # encode description
        self.gru.flatten_parameters()
        temp, lang_last = self.gru(lang_feat)
        print("lan input", lang_feat.data.shape)
        print("temp", temp.data.shape)
        print("lang_last",lang_last.shape)
        # print("output rnn temp data", temp.data.shape, "batch_size", temp.batch_sizes.shape, "sorted_indices", temp.sorted_indices.shape,"lang :", lang_last.shape)
        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size, hidden_size * num_dir
        print("lang_last after ",lang_last.shape)
        import sys
        sys.exit()
        # store the encoded language features
        data_dict["lang_emb"] = lang_last # B, hidden_size
        
        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        return data_dict

