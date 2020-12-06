#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ERNIE-ViL model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import pdb

import six
import numpy as np
import paddle.fluid as fluid

from model.vl_transformer_encoder import encoder, pre_process_layer


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


class ErnieVilModel(object):
    """
    main class for ERNIE-ViL model
    """
    def __init__(self,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 task_ids,
                 input_mask,
                 image_embeddings,
                 image_loc,
                 input_image_mask,
                 config,
                 predict_feature=False,
                 predict_class=True,
                 use_attr=False,
                 use_soft_label=True,
                 seed=0,
                 init_seed=0,
                 debug=False):
        
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        
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
        self._hidden_act = config['hidden_act']
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

        self.seed = seed
        self.init_seed = init_seed
        self.debug = debug
        self.debug_var = {}
        
        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'], seed=self.init_seed)

        self._build_model(src_ids, position_ids, sentence_ids, task_ids, input_mask, \
                image_embeddings, image_loc, input_image_mask)

    def _build_model(self, src_ids, position_ids, sentence_ids, task_ids, input_mask, \
            image_embeddings, image_loc, input_image_mask):
        # padding id in vocabulary must be set to 0
        _emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)

        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))

        sent_emb_out = fluid.layers.embedding(
            sentence_ids,
            size=[self._sent_types, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer))

        emb_out = _emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out

        if self.debug:
            self.debug_var['emb_out_0'] = emb_out
            # self.debug_var['position_emb_out'] = position_emb_out
            # self.debug_var['sent_emb_out'] = sent_emb_out
            # import pdb; pdb.set_trace()

        emb_out = pre_process_layer(
            emb_out, 'nd', self._prepostprocess_dropout, name='pre_encoder', seed=self.seed)
        
        if self.debug:
            self.debug_var['emb_out_1'] = emb_out

        # (-1, 80, 1) @ (-1, 80, 1).T --> (-1, 80, 80)
        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)

        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        image_embeddings = fluid.layers.fc(image_embeddings,
                                      self._v_emb_size,
                                      param_attr=fluid.ParamAttr(
                                            name="image_emb.w_0",
                                            initializer=self._param_initializer),
                                      bias_attr = "image_emb.b_0",
                                      num_flatten_dims = 2)
        loc_emb_out = fluid.layers.fc(image_loc,
                                      self._v_emb_size,
                                      param_attr=fluid.ParamAttr(
                                            name="image_loc.w_0",
                                            initializer=self._param_initializer),
                                      bias_attr = "image_loc.b_0",
                                      num_flatten_dims = 2)

        emb_vl_out = image_embeddings + loc_emb_out
        emb_vl_out = pre_process_layer(  
            emb_vl_out, 'nd', self._prepostprocess_dropout, name='vl_pre_encoder', seed=self.seed)

        self_attn_image_mask = fluid.layers.matmul(
            x=input_image_mask, y=input_image_mask, transpose_y=True)

        self_attn_image_mask = fluid.layers.scale(
            x=self_attn_image_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_image_mask = fluid.layers.stack(
            x=[self_attn_image_mask] * self._v_head, axis=1)
        n_head_self_attn_image_mask.stop_gradient = True

        self_attn_vl_mask = fluid.layers.matmul(
            x=input_image_mask, y=input_mask, transpose_y=True)
        # (-1, 100, 1) @ (-1, 80, 1)

        self_attn_vl_mask = fluid.layers.scale(
            x=self_attn_vl_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_vl_mask = fluid.layers.stack(
            x=[self_attn_vl_mask] * self._co_head, axis=1)
        n_head_self_attn_vl_mask.stop_gradient = True

        if self.debug:
            self.debug_var['emb_out'] = _emb_out
            self.debug_var['emb_vl_out'] = emb_vl_out
            self.debug_var['n_head_self_attn_mask'] = n_head_self_attn_mask
            self.debug_var['n_head_self_attn_image_mask'] = n_head_self_attn_image_mask
            self.debug_var['n_head_self_attn_vl_mask'] = n_head_self_attn_vl_mask

        self._enc_out, self._enc_vl_out = encoder(
            enc_input=emb_out,
            enc_vl_input=emb_vl_out,
            attn_bias=n_head_self_attn_mask,
            attn_image_bias=n_head_self_attn_image_mask,
            attn_vl_bias=n_head_self_attn_vl_mask,
            
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
            param_initializer=self._param_initializer,
            v_biattention_id = self._v_biattention_id,
            t_biattention_id = self._t_biattention_id,
            name='encoder',
            seed=self.seed)
        
        if self.debug:
            self.debug_var['_enc_out'] = self._enc_out
            self.debug_var['_enc_vl_out'] = self._enc_vl_out

    def get_sequence_output(self):
        """ 
        Return sequence output of all text and img tokens
        """
        return self._enc_out, self._enc_vl_out

    def get_pooled_output(self):
        """
        Get the first feature of each sequence for classification
        """
        text_cls_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])

        text_cls_feat = fluid.layers.cast(
            x=text_cls_feat, dtype=self._emb_dtype)

        text_cls_feat = fluid.layers.fc(
            input=text_cls_feat,
            size=self._co_emb_size,
            act="relu",
            param_attr=fluid.ParamAttr(
                name="pooled_fc_text.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc_text.b_0")

        image_cls_feat = fluid.layers.slice(
            input=self._enc_vl_out, axes=[1], starts=[0], ends=[1])

        image_cls_feat = fluid.layers.cast(
                x=image_cls_feat, dtype=self._emb_dtype)

        image_cls_feat = fluid.layers.fc(
            input=image_cls_feat,
            size=self._co_emb_size,
            act="relu",
            param_attr=fluid.ParamAttr(
                name="pooled_fc_image.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc_image.b_0")
        return text_cls_feat, image_cls_feat

    def get_match_score(self, text, image, dropout_rate=0.0, mode="mul"):
        """
        match score for text [cls] and image [img] tokens
        """
        if mode == "sum":
            emb_fuse = text + image
        elif mode == "mul":
            emb_fuse = text * image
        else:
            "current mode %s is not supported" % mode
            return
        if dropout_rate > 0.0:

            emb_fuse = fluid.layers.dropout(
                emb_fuse,
                self._attention_dropout,
                dropout_implementation="upscale_in_train",
                seed=self.seed)
        if self.debug:
            self.debug_var['emb_fuse'] = emb_fuse
        return emb_fuse


def init_pretraining_params(exe, pretraining_params_path, main_program):
    """
    init pretraining params without lr and step info
    """
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        """
        Check existed params
        """
        if not isinstance(var, fluid.framework.Parameter):
            return False
        print('exitsted: ', var.name)
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params)
    print("Load pretraining parameters from {}.".format(
        pretraining_params_path))


def test_ernie_vil():
    
    def model_input_data():
        max_seq_len = 80
        max_img_len = 40
        feature_size = 2048
        
        shapes = [
            [-1, max_seq_len, 1],  # src_id
            [-1, max_seq_len, 1],  # pos_id
            [-1, max_seq_len, 1],  # sent_id
            [-1, max_seq_len, 1],  # task_id
            [-1, max_seq_len, 1],  # input_mask
            [-1, max_img_len, feature_size],  # image_embedding
            [-1, max_img_len, 5],  # image_loc
            [-1, max_img_len, 1],  # image_mask
        ]
        # import pdb; pdb.set_trace()
        dtypes = ['int64', 'int64', 'int64', 'int64', 'float32', 'float32', 'float32', 'float32',
                'int64', 'int64', 'int64', 'float32']
        lod_levels = [0] * len(dtypes)
        names = [
            "src_ids", "pos_ids", "sent_ids", "task_ids", "input_mask", "image_embeddings",
            "image_loc", "image_mask", "labels", "q_ids", "task_index", "binary_labels"
        ]

        inputs_var = []
        for i in range(8):
            inputs_var.append(
                fluid.layers.data(name=names[i], shape=shapes[i], dtype=dtypes[i])
            )
        return inputs_var
    
    
    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    # NOTE: useing model config that has all dropout set to zero
    config_path = '/home/ron_zhu/Disk2/ernie/model-ernie-vil-base-VCR-task-pre-en/ernie_vil_config.base.dev.json'
    config = ErnieVilConfig(config_path)

    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            (src_ids, pos_ids, sent_ids, task_ids,
             input_mask, image_embeddings, image_loc, image_mask) = model_input_data()
            ernie_vil = ErnieVilModel(
                src_ids=src_ids,
                position_ids=pos_ids,
                sentence_ids=sent_ids,
                task_ids=task_ids,
                input_mask=input_mask,
                image_embeddings=image_embeddings,
                image_loc=image_loc,
                input_image_mask=image_mask,
                config=config,
                debug=True,
            )

            h_cls, h_img = ernie_vil.get_pooled_output()
            fusion_fea = ernie_vil.get_match_score(
                text=h_cls,
                image=h_img,
                dropout_rate=0.0,
                mode='mul'
            )
            out_dict = ernie_vil.debug_var
            out_dict['h_cls'] = h_cls
            out_dict['h_img'] = h_img
    
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    init_pretraining_params(
        exe,
        '/home/ron_zhu/Disk2/ernie/model-ernie-vil-base-VCR-task-pre-en/params',
        test_prog
    )

    seq_len = 80
    img_seq_len = 40
    img_feat_size = 2048

    np.random.seed(1234)
    feed_list = {
        "src_ids": np.random.randint(100, high=30000, size=[2, seq_len, 1], dtype=np.int64),
        "pos_ids": np.random.randint(0, high=512, size=[2, seq_len, 1], dtype=np.int64),
        "sent_ids": np.random.randint(0, high=1, size=[2, seq_len, 1], dtype=np.int64),
        "task_ids": np.zeros([2, seq_len, 1], dtype=np.int64),
        "input_mask": np.random.normal(size=[2, seq_len, 1]).astype(np.float32),
        "image_embeddings": np.random.normal(size=[2, img_seq_len, img_feat_size]).astype(np.float32),
        "image_loc": np.random.normal(size=[2, img_seq_len, 5]).astype(np.float32),
        "image_mask": np.random.normal(size=[2, img_seq_len, 1]).astype(np.float32),
    }

    output = exe.run(
        test_prog,
        feed=feed_list,
        fetch_list=list(out_dict.values())
    )
    name_to_val = {k: v for k, v in zip(out_dict.keys(), output)}
    for k, v in name_to_val.items():
        print(k)
        print(v.shape)
        print(v)
        print()
        if v.ndim == 2:
            print(v[0])
        elif v.ndim == 3:
            print(v[0, 0])
        elif v.ndim == 4:
            print(v[0, 0, 0])
        print('-' * 100)
    print(feed_list['pos_ids'][0, :, 0])
    print(feed_list['sent_ids'][0, :, 0])


if __name__ == "__main__":
    test_ernie_vil()
