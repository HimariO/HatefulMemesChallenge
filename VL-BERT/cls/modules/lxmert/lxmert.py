
import io
import json
import wget
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL.Image
from transformers import LxmertForQuestionAnswering, LxmertForPreTraining, LxmertTokenizer, LxmertConfig
from transformers import AutoModel

from common.module import Module
from external.pytorch_pretrained_bert.modeling import BertPredictionHeadTransform, BertLayerNorm, gelu, GELU
from cls.modules.lxmert.processing_image import Preprocess
from cls.modules.lxmert.modeling_frcnn import GeneralizedRCNN
from cls.modules.lxmert.utils import Config
from cls.modules.lxmert import utils


# URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg",
URL = "https://vqa.cloudcv.org/media/test2014/COCO_test2014_000000262567.jpg"
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"


def build_image_encoder():
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
    return frcnn.backbone, frcnn.roi_heads


class LXMERT(Module):

    def __init__(self, dummy_config):
        super(LXMERT, self).__init__(dummy_config)
        
        frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        # self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
        self.backbone, self.roi_heads = build_image_encoder()
        self.lxmert_vqa = LxmertForPreTraining.from_pretrained("unc-nlp/lxmert-base-uncased")
        # self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")
        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.image_preprocess = Preprocess(frcnn_cfg)
        
        hid_dim = self.lxmert_vqa.config.hidden_size
        # transform = BertPredictionHeadTransform(self.config.NETWORK.VLBERT)

        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            GELU(),
            BertLayerNorm(hid_dim),
            nn.Dropout(self.config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
            nn.Linear(hid_dim, self.config.NETWORK.CLASSIFIER_CLASS),
        )

    def prepare_text(self, question, question_tags, question_mask):
        batch_size, max_q_len = question.shape
        max_len = question_mask.sum(1).max() + 2
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        q_end = 1 + question_mask.sum(1, keepdim=True)
        input_ids = torch.zeros((batch_size, max_len), dtype=question.dtype, device=question.device)
        input_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=question.device)
        
        input_type_ids = torch.zeros((batch_size, max_len), dtype=question.dtype, device=question.device)
        text_tags = input_type_ids.new_zeros((batch_size, max_len))
        grid_i, grid_j = torch.meshgrid(torch.arange(batch_size, device=question.device),
                                        torch.arange(max_len, device=question.device))

        input_mask[grid_j > q_end] = 0
        # input_type_ids[(grid_j > q_end) & (grid_j <= a_end)] = 1
        q_input_mask = (grid_j > 0) & (grid_j < q_end)
        sep_idx = (question == sep_id).nonzero()
        for index in sep_idx:
            input_type_ids[index[0], index[1] + 1:] = self.config.NETWORK.VLBERT.visual_tag_type

        input_ids[:, 0] = cls_id
        input_ids[grid_j == q_end] = sep_id

        input_ids[q_input_mask] = question[question_mask]
        text_tags[q_input_mask] = question_tags[question_mask]

        return input_ids, input_type_ids, text_tags, input_mask
    
    def extract_image_feat(self, image, boxes):
        features = self.backbone(image)
        features = [features[f] for f in self.roi_heads.in_features]
        box_list = [box for box in boxes]
        box_features = self.roi_heads._shared_roi_transform(
            features, boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x
        # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
        features_list = feature_pooled.split([len(bb) for bb in boxes])
        return torch.stack(features_list, dim=0)
    
    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      text_ids,
                      img_boxes,
                      text_tags,
                      label: torch.Tensor,
                      *sample_id_and_more,
                      loss_fn=F.binary_cross_entropy_with_logits,
                      ):        
        # inputs = lxmert_tokenizer(
        #     text,
        #     padding="max_length",
        #     max_length=20,
        #     truncation=True,
        #     return_token_type_ids=True,
        #     return_attention_mask=True,
        #     add_special_tokens=True,
        #     return_tensors="pt"
        # )
        
        text_mask = (text_ids > 0.5)
        (text_input_ids, text_token_type_ids,
        text_tags, text_mask) = self.prepare_text(text_ids, text_tags, text_mask)

        # images, sizes, scales_yx = self.image_preprocess(image)
        # output_dict = self.frcnn(
        #     images, 
        #     sizes, 
        #     scales_yx=scales_yx, 
        #     padding="max_detections",
        #     max_detections=30,
        #     return_tensors="pt",
        # )
        # normalized_boxes = output_dict.get("normalized_boxes")
        # features = output_dict.get("roi_features")

        box_mask = (boxes[:, :, 0] > - 1.5)
        boxes = boxes[..., :4]  # NOTE: (B, N, 5) -> (B, N, 4) remove class id
        features = self.extract_image_feat(image, boxes)
        im_size = torch.unsqueeze(torch.cat([im_info[:, :2], im_info[:, :2]], dim=1), dim=1)
        normalized_boxes = boxes / im_size

        lxmert_output = self.lxmert_vqa.lxmert(
            input_ids=text_input_ids,
            attention_mask=text_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            visual_attention_mask=box_mask,
            token_type_ids=text_token_type_ids,
            return_dict=True,
            output_attentions=False,
        )
        pooled_output = lxmert_output[2]
        logits = self.logit_fc(pooled_output)
        
        if label.ndim == 1:
            label = label.unsqueeze(1)
        # loss
        ans_loss = loss_fn(logits, label) * label.size(1)
        loss = ans_loss.mean()
        
        outputs = {
            'label_logits': logits,
            'label': label,
            'ans_loss': ans_loss
        }
        return outputs, loss
    
    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          text_ids,
                          img_boxes,
                          text_tags,
                          *args):
        
        text_mask = (text_ids > 0.5)
        (text_input_ids, text_token_type_ids,
        text_tags, text_mask) = self.prepare_text(text_ids, text_tags, text_mask)

        # images, sizes, scales_yx = self.image_preprocess(image)
        # output_dict = self.frcnn(
        #     images, 
        #     sizes, 
        #     scales_yx=scales_yx, 
        #     padding="max_detections",
        #     max_detections=30,
        #     return_tensors="pt",
        # )
        # normalized_boxes = output_dict.get("normalized_boxes")
        # features = output_dict.get("roi_features")
        
        box_mask = (boxes[:, :, 0] > - 1.5)
        boxes = boxes[..., :4]
        features = self.extract_image_feat(image, boxes)
        im_size = torch.unsqueeze(torch.cat([im_info[:, :2], im_info[:, :2]], dim=1), dim=1)
        normalized_boxes = boxes / im_size

        lxmert_output = self.lxmert_vqa.lxmert(
            input_ids=text_input_ids,
            attention_mask=text_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=text_token_type_ids,
            return_dict=True,
            output_attentions=False,
        )
        pooled_output = lxmert_output[2]
        logits = self.logit_fc(pooled_output)

        outputs = {
            'label_logits': logits,
        }
        return outputs


class LXMERT_EXTRACTER:
    def __init__(self, device='cuda:0'):
        self.device = device
        # load models and model components
        frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg).to(device)

        self.image_preprocess = Preprocess(frcnn_cfg)
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased").to(device)

    def __call__(self, text, img, img_tags=None):
        if img_tags is not None:
            text += F" [SEP] {img_tags}"
        
        images, sizes, scales_yx = self.image_preprocess(img)
        output_dict = self.frcnn(
            images.to(self.device), 
            sizes, 
            scales_yx=scales_yx, 
            padding="max_detections",
            max_detections=30,
            return_tensors="pt"
        )
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")

        inputs = self.lxmert_tokenizer(
            text,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        lxmert_output = self.lxmert_gqa.lxmert(
            input_ids=inputs.input_ids.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device),
            visual_feats=features.to(self.device),
            visual_pos=normalized_boxes.to(self.device),
            token_type_ids=inputs.token_type_ids.to(self.device),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )
        pooled_output = lxmert_output[2]
        return pooled_output
    
    def get_img_feat(self, img):
        images, sizes, scales_yx = self.image_preprocess(img)
        output_dict = self.frcnn(
            images.to(self.device), 
            sizes, 
            scales_yx=scales_yx, 
            padding="max_detections",
            max_detections=30,
            return_tensors="pt"
        )
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")
        return features