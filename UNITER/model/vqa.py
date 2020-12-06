"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VQA model
"""
from collections import defaultdict

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from apex import amp
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel, VillaModel
from .ot import optimal_transport_dist


class UniterForITM(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.itm_output_dropout = nn.Dropout(0.1)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True, ot_weight=0.1):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        answer_scores = self.itm_output(self.itm_output_dropout(pooled_output))

        if compute_loss:
            targets = batch['targets']
            targets = (targets > 0.5).long()
            targets = torch.abs(targets - 1)
            targets = torch.squeeze(targets, dim=-1)
            # NOTE: 0: hateful, 1: normal reverse classes order from usal
            itm_loss = F.cross_entropy(answer_scores, targets, reduction='none')
            
            return itm_loss
        else:
            return answer_scores


class UniterForVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)

        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        # self.vqa_output = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size*2),
        #     GELU(),
        #     LayerNorm(config.hidden_size*2, eps=1e-12),
        #     nn.Linear(config.hidden_size*2, num_answer)
        # )
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True, ot_weight=0.1):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        answer_scores = self.vqa_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            vqa_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets, reduction='none')
            
            if 'ot_inputs' in batch:
                ot_inputs = batch['ot_inputs']
                bs = img_feat.shape[0]
                dummy_target = torch.ones([bs, 1]).int().to(img_feat.device)
                itm_loss = self.ot_loss(
                    ot_inputs, input_ids, img_feat, sequence_output, dummy_target)
                ot_loss = itm_loss[0].view([bs, bs]).sum(dim=-1, keepdim=True)
                vqa_loss += ot_loss * ot_weight
            return vqa_loss
        else:
            return answer_scores
    
    def ot_loss(self, ot_inputs, input_ids, img_feat, sequence_output, itm_targets):
        # OT loss
        if ot_inputs is not None:
            ot_scatter = ot_inputs['ot_scatter']

            b = sequence_output.size(0)
            tl = input_ids.size(1)
            il = img_feat.size(1)
            max_l = max(ot_inputs['scatter_max'] + 1, tl+il)

            ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
            ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                                  dtype=sequence_output.dtype,
                                  device=sequence_output.device
                                  ).scatter_(dim=1, index=ot_scatter,
                                             src=sequence_output)
            txt_emb = ctx_emb[:, :tl, :]
            img_emb = ctx_emb[:, tl:tl+il, :]

            txt_pad = ot_inputs['txt_pad']
            img_pad = ot_inputs['img_pad']
            # NOTE: run in fp32 for stability
            ot_dist = optimal_transport_dist(txt_emb.float(), img_emb.float(),
                                             txt_pad, img_pad).to(txt_emb)
            ot_pos_dist = ot_dist.masked_select(itm_targets == 1)
            ot_neg_dist = ot_dist.masked_select(itm_targets == 0)
            ot_loss = (ot_pos_dist, ot_neg_dist)
        else:
            ot_loss = None
        return ot_loss


class VillaForVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.uniter = VillaModel(config, img_dim)
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        self.apply(self.init_weights)
        
        self.img_noise = None
        self.txt_noise = None
        self.kl = nn.KLDivLoss(reduction='mean')
        self.adv_grad_scale = 0.1
    
    def f_norm(self, tensor_3d):
        og_type = tensor_3d.dtype
        fp32_norm = torch.norm(tensor_3d.float(), p='fro', dim=(1, 2))
        return fp32_norm.to(og_type)
    
    def fgsm(self, gradz, step_size):
        # assert gradz.dim() == 3
        # denom = self.f_norm(gradz).unsqueeze(-1).unsqueeze(-1)
        # norm_gradz = gradz / denom
        # return step_size * gradz
        return step_size * torch.sign(gradz)
        # return step_size * norm_gradz
    
    def f_norm_proj(self, mtxs, bound=1e-3):
        """
        Rescale matrics to bounded Frobenius norm
        (clip_by_norm in tensorflow)
        """
        assert mtxs.dim() == 3
        # _pw_mtxs = (mtxs ** 2 + 1e-6).sum(dim=(1, 2))
        # _norm_per_sample = torch.sqrt(_pw_mtxs)
        norm_per_sample = self.f_norm(mtxs)
        one = torch.tensor(1.0).to(mtxs.dtype).to(norm_per_sample.device)
        sample_scale = torch.where(
            norm_per_sample > bound,
            bound / norm_per_sample,
            one
        )
        sample_scale = sample_scale.unsqueeze(-1).unsqueeze(-1)
        return mtxs * sample_scale
    
    def apply_kl(self, bin_logit_a, bin_logit_b):
        pred_a = torch.sigmoid(bin_logit_a)
        prob_a = torch.cat([pred_a, 1- pred_a], dim=-1).log()
        pred_b = torch.sigmoid(bin_logit_b)
        prob_b = torch.cat([pred_b, 1- pred_b], dim=-1)
        return self.kl(prob_a, prob_b)
    
    def forward(self, batch, compute_adv=False, compute_loss=True,
                optim=None, clip_norm=1e-0, fgsm_step=1e-2, kl_weight=1.0):
        if compute_adv:
            return self.forward_adv(
                batch, compute_loss=compute_loss, optim=optim,
                clip_norm=clip_norm, fgsm_step=fgsm_step, kl_weight=kl_weight)
        else:
            return self._forward(
                batch, compute_loss=compute_loss)

    def _forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        answer_scores = self.vqa_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            vqa_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets, reduction='none')
            return vqa_loss
        else:
            return answer_scores

    def forward_adv(self, batch, optim=None, compute_loss=True, clip_norm=1e-0, fgsm_step=1e-2, kl_weight=1.0):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']

        """
        Normal forward pass
        """
        embeds, extend_atten = self.uniter(input_ids, position_ids,
                                            img_feat, img_pos_feat,
                                            attn_masks, embed_only=True)
        sequence_output = self.uniter.forward_encode(embeds, extend_atten,
                                                     gather_index=gather_index,
                                                     output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        answer_scores = self.vqa_output(pooled_output)

        """
        Calc adversial noise on txt and img embedding
        """
        
        txt_embed = embeds['txt']
        img_embed = embeds['img']
        self.img_noise = torch.zeros_like(img_embed).uniform_(-fgsm_step / 10, fgsm_step / 10)
        self.txt_noise = torch.zeros_like(txt_embed).uniform_(-fgsm_step / 10, fgsm_step / 10)
        img_noise = Variable(self.img_noise, requires_grad=True)
        txt_noise = Variable(self.txt_noise, requires_grad=True)
        
        embeds_rand_noise = {
            'txt': txt_embed + txt_noise,
            'img': img_embed + img_noise,
        }
        noisy_seq_output = self.uniter.forward_encode(embeds_rand_noise, extend_atten,
                                                     gather_index=gather_index,
                                                     output_all_encoded_layers=False)
        noisy_pooled_output = self.uniter.pooler(noisy_seq_output)
        noisy_scores = self.vqa_output(noisy_pooled_output)
        
        targets = batch['targets']
        adv_ce_loss = F.binary_cross_entropy_with_logits(
                noisy_scores, targets, reduction='mean') * self.adv_grad_scale
        adv_kl_regular = self.apply_kl(noisy_scores, answer_scores) * self.adv_grad_scale
        adv_loss = adv_ce_loss

        with amp.scale_loss(adv_loss, optim) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        # adv_loss.backward()
        
        img_pert = self.fgsm(img_noise.grad, fgsm_step)
        txt_pert = self.fgsm(txt_noise.grad, fgsm_step)
        self.img_noise = self.f_norm_proj(self.img_noise + img_pert.data, bound=clip_norm*2)
        self.txt_noise = self.f_norm_proj(self.txt_noise + txt_pert.data, bound=clip_norm)
        # import pdb; pdb.set_trace()

        """
        Inference on txt and img adversial example
        """

        embeds_adv_img = {
            'txt': txt_embed,
            'img': img_embed + self.img_noise,
        }
        advi_seq_output = self.uniter.forward_encode(embeds_adv_img, extend_atten,
                                                     gather_index=gather_index,
                                                     output_all_encoded_layers=False)
        advi_pooled_output = self.uniter.pooler(advi_seq_output)
        advi_scores = self.vqa_output(advi_pooled_output)
        adv_img_ce_loss = F.binary_cross_entropy_with_logits(advi_scores, targets, reduction='mean')
        adv_img_kl_loss = self.apply_kl(advi_scores, answer_scores) * kl_weight
        adv_img_loss = adv_img_ce_loss + adv_img_kl_loss
        # print('[IMG] ', adv_img_ce_loss, adv_img_kl_loss)
        
        embeds_adv_txt = {
            'txt': txt_embed + self.txt_noise,
            'img': img_embed,
        }
        advt_seq_output = self.uniter.forward_encode(embeds_adv_txt, extend_atten,
                                                     gather_index=gather_index,
                                                     output_all_encoded_layers=False)
        advt_pooled_output = self.uniter.pooler(advt_seq_output)
        advt_scores = self.vqa_output(advt_pooled_output)

        adv_txt_ce_loss = F.binary_cross_entropy_with_logits(advt_scores, targets, reduction='mean')
        adv_txt_kl_loss = self.apply_kl(advt_scores, answer_scores) * kl_weight
        adv_txt_loss = adv_txt_ce_loss + adv_txt_kl_loss
        # print('[TXT] ', adv_txt_ce_loss, adv_txt_kl_loss)
        
        final_loss = F.binary_cross_entropy_with_logits(answer_scores, targets, reduction='mean')
        final_loss += adv_img_loss + adv_txt_loss
        # print('[Final] ', final_loss.cpu().item())
        # print('advt_scores: ', torch.abs(targets - torch.sigmoid(advt_scores)))
        # print('advi_scores: ', torch.abs(targets - torch.sigmoid(advi_scores)))
        # print('answer_scores: ', torch.abs(targets - torch.sigmoid(answer_scores)))
        # print('targets: ', targets)
        # print('-' * 100)
        # import pdb; pdb.set_trace()

        return final_loss

