import json
import pdb
from pytorch_lightning.trainer import optimizers

import six
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model.pt_vl_transformer_encoder import encoder, pre_process_layer
from utils.pl_metric import LitAUROC
from utils.scheduler import GradualWarmupScheduler


class ErnieVilConfig(object):
    """
    configuration for ernie-vil
    """

    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]
    
    def print_config(self):
        """
        print configuration value
        """
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class ErnieVilModel(nn.Module):
    """
    main class for ERNIE-ViL model
    """

    def __init__(self,
                 config,
                 predict_feature=False,
                 predict_class=True,
                 use_attr=False,
                 use_soft_label=True,
                 fusion_method="mul",
                 fusion_dropout=0.1,
                ):
        super().__init__()
        self.fusion_method = fusion_method
        self.fusion_dropout = fusion_dropout
        hidden_act_map = {
            'gelu': F.gelu,
        }
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']

        self._v_feat_size = 2048
        self._v_head = config['v_num_attention_heads']
        self._v_emb_size = config['v_hidden_size']
        self._v_inter_hid = config['v_intermediate_size']

        self._co_head = config['co_num_attention_heads']
        self._co_emb_size = config['co_hidden_size']
        self._co_inter_hid = config['co_intermediate_size']

        self._voc_size = config['vocab_size']
        self._class_size = config['class_size']
        self._class_attr_size = config['class_attr_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._sent_types = config['sent_type_vocab_size']
        self._task_types = config['task_type_vocab_size']
        self._hidden_act = hidden_act_map[config['hidden_act']]
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._v_biattention_id = config['v_biattention_id']
        self._t_biattention_id = config['t_biattention_id']

        self._predict_feature = predict_feature
        self._predict_class = predict_class
        self._use_attr = use_attr
        self._use_soft_label = use_soft_label
        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._image_emb_name = "image_embedding"
        self._loc_emb_name = "loc_embedding"
        self._dtype = "float32"
        self._emb_dtype = "float32"

        self._build_model()
    
    def load_paddle_weight(self, w_dict):
        translate_name = {
            nn.Embedding: {
                'weight': '',
            },
            nn.Linear: {
                'weight': 'w_0',
                'bias': 'b_0',
            }
        }

        new_state_dict = {}
        for mn, md in self.named_children():
            if hasattr(md, 'load_paddle_weight'):
                print('v' * 10)
                md.load_paddle_weight(w_dict, "")
                print('^' * 10)
            else:
                # Native pytorch nn modules
                for n, p in md.named_parameters():
                    blocks = [
                        translate_name[type(md)][sn]
                        if type(md) in translate_name
                        else sn
                        for sn in n.split('.')
                    ]
                    new_p_name = '.'.join(blocks)
                    pd_full_name = '.'.join([mn, new_p_name]) if new_p_name else mn
                    pt_full_name = f"{mn}.{n}"
                    
                    if pd_full_name in w_dict:
                        new_state_dict[pt_full_name] = torch.tensor(w_dict[pd_full_name])
                        if 'weight' in pt_full_name and isinstance(md, nn.Linear):
                            new_state_dict[pt_full_name] = new_state_dict[pt_full_name].T
                        print(f'matched: {pd_full_name} -> {pt_full_name}', new_state_dict[pt_full_name].shape)
                    else:
                        print('not match: ', pd_full_name)
                        import pdb; pdb.set_trace()
        mismatchs = self.load_state_dict(new_state_dict, strict=False)

    def _build_model(self, ):
        self.word_embedding = nn.Embedding(self._voc_size, self._emb_size)
        self.pos_embedding = nn.Embedding(self._max_position_seq_len, self._emb_size)
        self.sent_embedding = nn.Embedding(self._sent_types, self._emb_size)

        self.pre_encoder = pre_process_layer(
            'nd',
            self._emb_size,
            dropout_rate=self._prepostprocess_dropout,
            postfix='pre_encoder',
        )
        self.image_emb = nn.Linear(self._v_feat_size, self._v_emb_size, bias=True)
        self.image_loc = nn.Linear(5, self._v_emb_size, bias=True)
        self.vl_pre_encoder = pre_process_layer(
            'nd',
            self._v_emb_size,
            dropout_rate=self._prepostprocess_dropout,
            postfix='vl_pre_encoder',
        )

        self.encoder = encoder(
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            v_head=self._v_head,
            v_key=self._v_emb_size // self._v_head,
            v_value=self._v_emb_size // self._v_head,
            v_model=self._v_emb_size,
            v_inner_hid=self._v_inter_hid,
            co_head=self._co_head,
            co_key=self._co_emb_size // self._co_head,
            co_value=self._co_emb_size // self._co_head,
            co_model=self._co_emb_size,
            co_inner_hid=self._co_inter_hid,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            v_biattention_id=self._v_biattention_id,
            t_biattention_id=self._t_biattention_id,
            name='encoder',
        )

        self.pooled_fc_text = nn.Linear(self._emb_size, self._co_emb_size, bias=True)
        self.pooled_fc_image = nn.Linear(self._v_emb_size, self._co_emb_size, bias=True)
        self.emb_fuse_dropout = nn.Dropout(self.fusion_dropout)
    
    def get_pooled_output(self, _enc_out, _enc_vl_out):
        text_cls_feat = _enc_out[:, 0, :]
        text_cls_feat = self.pooled_fc_text(text_cls_feat)
        text_cls_feat = torch.relu(text_cls_feat)
        
        image_cls_feat = _enc_vl_out[:, 0, :]
        image_cls_feat = self.pooled_fc_image(image_cls_feat)
        image_cls_feat = torch.relu(image_cls_feat)
        return text_cls_feat, image_cls_feat
    
    def get_match_score(self, text, image, mode="mul"):
        if mode == "sum":
            emb_fuse = text + image
        elif mode == "mul":
            emb_fuse = text * image
        else:
            raise ValueError(f"current mode {mode} is not supported")
        emb_fuse = self.emb_fuse_dropout(emb_fuse)
        return emb_fuse
    
    def forward(self,                  
                src_ids,
                position_ids,
                sentence_ids,
                task_ids,
                input_mask,
                image_embeddings,
                image_loc,
                input_image_mask,
                pooled_output=False,
                match_score=False,):
        emb_out = self.word_embedding(src_ids)
        position_emb_out = self.pos_embedding(position_ids)
        sent_emb_out = self.sent_embedding(sentence_ids)

        emb_out = emb_out + position_emb_out
        emb_out = emb_out_0 = emb_out + sent_emb_out
        emb_out = emb_out_1 = self.pre_encoder(emb_out)

        self_attn_mask = torch.matmul(input_mask, input_mask.permute([0, 2, 1]))
        self_attn_mask = (self_attn_mask - 1.0) * 10000.0
        n_head_self_attn_mask = torch.stack([self_attn_mask] * self._n_head, dim=1)
        n_head_self_attn_mask = n_head_self_attn_mask.detach()

        image_embeddings = self.image_emb(image_embeddings)
        loc_emb_out = self.image_loc(image_loc)
        emb_vl_out = image_embeddings + loc_emb_out
        emb_vl_out = self.vl_pre_encoder(emb_vl_out)

        self_attn_image_mask = torch.matmul(
            input_image_mask,
            input_image_mask.permute([0, 2, 1])
        )
        self_attn_image_mask = (self_attn_image_mask - 1.0) * 10000.0
        n_head_self_attn_image_mask = torch.stack([self_attn_image_mask] * self._v_head, dim=1)
        n_head_self_attn_image_mask = n_head_self_attn_image_mask.detach()
        
        self_attn_vl_mask = torch.matmul(
            input_image_mask,
            input_mask.permute([0, 2, 1])
        )
        self_attn_vl_mask = (self_attn_vl_mask - 1.0) * 10000.0
        n_head_self_attn_vl_mask = torch.stack([self_attn_vl_mask] * self._co_head, dim=1)
        n_head_self_attn_vl_mask = n_head_self_attn_vl_mask.detach()

        enc_out, enc_vl_out = self.encoder(
            emb_out,
            emb_vl_out,
            n_head_self_attn_mask,
            n_head_self_attn_image_mask,
            n_head_self_attn_vl_mask,
        )

        if match_score:
            h_cls, h_img = self.get_pooled_output(enc_out, enc_vl_out)
            emb_fuse = self.get_match_score(h_cls, h_img, mode=self.fusion_method)
            return emb_fuse
        elif pooled_output:
            return self.pooled_output(enc_out, enc_vl_out)
        else:
            return enc_out, enc_vl_out


class LitErnieVil(pl.LightningModule):

    def __init__(self, args, fusion_method="mul", fusion_dropout=0.1, cls_head='linear'):
        super().__init__()
        hparams = vars(args)
        hparams.update({
            "fusion_method": fusion_method,
            "fusion_dropout": fusion_dropout,
            "cls_head": cls_head,
        })
        self.hparams = hparams
        self.args = args
        self.train_accuracy = pl.metrics.classification.Accuracy()
        self.val_accuracy = pl.metrics.classification.Accuracy()
        self.val_auroc = LitAUROC()

        self.ernie_config = ErnieVilConfig(args.ernie_config_path)
        self.ernie_vil = ErnieVilModel(
            self.ernie_config,
            fusion_dropout=fusion_dropout,
            fusion_method=fusion_method,
        )

        if cls_head == 'linear':
            self.fc = nn.Sequential(
                # nn.Linear(self.ernie_vil._co_emb_size, self.ernie_vil._co_emb_size * 2),
                nn.Linear(self.ernie_vil._co_emb_size, 2, bias=True),
            )
            torch.nn.init.normal_(self.fc[0].weight, mean=0.0, std=0.02)
            # torch.nn.init.xavier_normal_(self.fc[0].weight)
            # torch.nn.init.constant_(self.fc[0].bias, 0.0)
        elif cls_head == 'mlm':
            self.fc = nn.Sequential(
                nn.Linear(self.ernie_vil._co_emb_size, self.ernie_vil._co_emb_size),
                nn.GELU(),
                nn.LayerNorm(self.ernie_vil._co_emb_size),
                nn.Dropout(fusion_dropout),
                nn.Linear(self.ernie_vil._co_emb_size, 2),
            )
        else:
            raise ValueError(f'cls_head: {cls_head} is not supported!')
    
    def load_paddle_weight(self, npz_path):
        w_dict = np.load(npz_path)
        self.ernie_vil.load_paddle_weight(w_dict)
    
    def forward(self,
                src_ids,
                position_ids,
                sentence_ids,
                task_ids,
                input_mask,
                image_embeddings,
                image_loc,
                input_image_mask):
        emb_fuse = self.ernie_vil(
            src_ids,
            position_ids,
            sentence_ids,
            task_ids,
            input_mask,
            image_embeddings,
            image_loc,
            input_image_mask,
            match_score=True,
        )
        cls_logit = self.fc(emb_fuse)
        return cls_logit
    
    def training_step(self, batch, batch_idx):
        (src_ids, src_pos, src_seg, src_task, src_masks,
        image_embeddings, image_loc, image_mask, labels, batch_anno_ids,
        _, _, _) = batch
        logits = self.forward(
            src_ids,
            src_pos,
            src_seg,
            src_task,
            src_masks,
            image_embeddings,
            image_loc,
            image_mask
        )

        if labels.ndim == 2 and labels.dtype == torch.long:
            labels = torch.squeeze(labels, dim=-1)
        loss = F.cross_entropy(logits, labels).mean()
        
        self.log('train_loss', loss)
        self.log('train_acc_step', self.train_accuracy(logits, labels))

        if hasattr(self, "scheduler"):
            lr = torch.tensor(self.scheduler.get_last_lr()[0])
            self.log('lr', lr, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (src_ids, src_pos, src_seg, src_task, src_masks,
         image_embeddings, image_loc, image_mask, labels, batch_anno_ids,
         _, _, _) = batch
        logits = self.forward(
            src_ids,
            src_pos,
            src_seg,
            src_task,
            src_masks,
            image_embeddings,
            image_loc,
            image_mask
        )

        pred = F.softmax(logits, dim=-1)
        self.val_auroc(pred[..., 1], labels)
        self.val_accuracy(pred, labels)
        return {
            'predict': pred, 
            'anno_idx': batch_anno_ids
        }

    def validation_epoch_end(self, validation_step_outputs):
        self.log('val_auroc_epoch', self.val_auroc.compute())
        self.log('val_aucc_epoch', self.val_accuracy.compute())
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, test_outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        T_0 = self.args.num_train_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0, T_mult=1, eta_min=1e-8)
        scheduler_warmup = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=self.args.warmup_steps,
            after_scheduler=scheduler
        )
        self.scheduler = scheduler_warmup
        return [optimizer], [{
            'scheduler': scheduler_warmup,
            'interval': 'step',
        }]

def test_ernie_vil():
    w_dict = np.load('/home/ron_zhu/Disk2/ernie/ernie-vil-base-vcr.npz')
    # NOTE: using config that set all dropout to zero
    config_path = '/home/ron_zhu/Disk2/ernie/model-ernie-vil-base-VCR-task-pre-en/ernie_vil_config.base.dev.json'
    config = ErnieVilConfig(config_path)
    ernie_vil = ErnieVilModel(config, fusion_dropout=0.0)
    ernie_vil.load_paddle_weight(w_dict)
    ernie_vil.eval()

    seq_len = 80
    img_seq_len = 40
    img_feat_size = 2048

    np.random.seed(1234)
    src_ids = np.random.randint(100, high=30000, size=[2, seq_len, 1], dtype=np.int64)[..., 0]
    position_ids = np.random.randint(0, high=512, size=[2, seq_len, 1], dtype=np.int64)[..., 0]
    sentence_ids = np.random.randint(0, high=1, size=[2, seq_len, 1], dtype=np.int64)[..., 0]
    task_ids = None
    input_mask = np.random.normal(size=[2, seq_len, 1])
    image_embeddings = np.random.normal(size=[2, img_seq_len, img_feat_size])
    image_loc = np.random.normal(size=[2, img_seq_len, 5])
    input_image_mask = np.random.normal(size=[2, img_seq_len, 1])
    print(src_ids[0])
    
    src_ids = torch.tensor(src_ids)
    position_ids = torch.tensor(position_ids)
    sentence_ids = torch.tensor(sentence_ids)
    input_mask = torch.tensor(input_mask).float()
    image_embeddings = torch.tensor(image_embeddings).float()
    image_loc = torch.tensor(image_loc).float()
    input_image_mask = torch.tensor(input_image_mask).float()

    emb_fuse = ernie_vil(
        src_ids,
        position_ids,
        sentence_ids,
        task_ids,
        input_mask,
        image_embeddings,
        image_loc,
        input_image_mask,
        match_score=True,
    )
    
    print(emb_fuse)
    print(emb_fuse[0])
    print(emb_fuse.shape)


if __name__ == "__main__":
    from loguru import logger
    with logger.catch():
        test_ernie_vil()
