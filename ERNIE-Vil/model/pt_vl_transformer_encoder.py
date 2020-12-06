import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class multi_head_attention(nn.Module):

    def __init__(self, 
                d_key,
                d_value,
                d_model,
                hidden_size,
                n_head=1,
                dropout_rate=0.,
                postfix='_multi_head_att'):
        super().__init__()
        self.postfix = postfix
        self.n_head = n_head
        self.hidden_size = hidden_size

        self.query_fc = nn.Linear(d_model, d_key * n_head, bias=True)
        self.key_fc = nn.Linear(hidden_size, d_key * n_head, bias=True)
        self.value_fc = nn.Linear(hidden_size, d_value * n_head, bias=True)
        self.drop_out = nn.Dropout(dropout_rate)  # pytorch default run upscale_in_train

        self.output_fc = nn.Linear(max(d_model, hidden_size), d_model, bias=True)
    
    def __compute_qkv(self, queries, keys, values):
        b, n, c = queries.shape
        q = self.query_fc(queries.view(b * n, c))
        q = q.view([b, n, self.n_head, -1])
        q = q.permute([0, 2, 1, 3])
        
        b, n, c = keys.shape
        k = self.key_fc(keys.view(b * n, c))
        k = k.view([b, n, self.n_head, -1])
        k = k.permute([0, 2, 1, 3])
        
        b, n, c = values.shape
        v = self.value_fc(values.view(b * n, c))
        v = v.view([b, n, self.n_head, -1])
        v = v.permute([0, 2, 1, 3])
        
        return q, k, v
    
    def scaled_dot_product_attention(self, q, k, v, attn_bias):
        d_key = k.shape[-1]
        scaled_q = q * (d_key ** -0.5)
        product = torch.matmul(scaled_q, k.permute(0, 1, 3, 2))
        if attn_bias is not None:
            if product.shape != attn_bias.shape:
                import pdb; pdb.set_trace()
            product += attn_bias
        weights = torch.softmax(product, dim=-1)
        weights = self.drop_out(weights)
        out = torch.matmul(weights, v)
        return out
    
    def __combine_heads(self, x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = x.permute([0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return trans_x.reshape(trans_x.shape[:2] + (-1,))
    
    def forward(self, queries, keys, values, attn_bias,):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q, k, v = self.__compute_qkv(queries, keys, values)
        
        ctx_multiheads = self.scaled_dot_product_attention(q, k, v, attn_bias)
        out = self.__combine_heads(ctx_multiheads)
        
        b, n, c = out.shape
        proj_out = self.output_fc(out)
        return proj_out
    
    def load_paddle_weight(self, w_dict, name_prefix):
        padd_module_name = name_prefix + self.postfix
        translate_name = {
            'weight': 'w_0',
            'bias': 'b_0',
        }

        new_state_dict = {}
        for n, p in self.named_parameters():
            blocks = [
                translate_name[nn] if nn in translate_name else nn
                for nn in n.split('.')
            ]
            new_p_name = '.'.join(blocks)
            full_name = padd_module_name + '_' + new_p_name
            if full_name in w_dict:
                new_state_dict[n] = torch.tensor(w_dict[full_name])
                if 'weight' in n:
                    new_state_dict[n] = new_state_dict[n].T
                print('matched: ', f"{full_name} --> {n}", new_state_dict[n].shape)
            else:
                print('not match: ', full_name)
                import pdb; pdb.set_trace()
        self.load_state_dict(new_state_dict)


class positionwise_feed_forward(nn.Module):

    def __init__(self, d_inner_hid,
                        d_hid,
                        dropout_rate,
                        hidden_act_fn=None,
                        postfix='_ffn') -> None:
        super().__init__()
        self.fc_0 = nn.Linear(d_hid, d_inner_hid, bias=True)
        self.fc_1 = nn.Linear(d_inner_hid, d_hid, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_act_fn = hidden_act_fn
        self.postfix = postfix

    def forward(self, x):
        """
        x: (-1, 80, 1024)
        hidden: (-1, 80, 4096)
        out: (-1, 80, 1024)
        """
        b, n, c = x.shape
        hidden = self.fc_0(x)
        if self.hidden_act_fn is not None:
            hidden = self.hidden_act_fn(hidden)
        hidden = self.dropout(hidden)
        out = self.fc_1(hidden)
        return out
    
    def load_paddle_weight(self, w_dict, name_prefix):
        padd_module_name = name_prefix + self.postfix
        translate_name = {
            'weight': 'w_0',
            'bias': 'b_0',
        }

        new_state_dict = {}
        for n, p in self.named_parameters():
            blocks = [
                translate_name[nn] if nn in translate_name else nn
                for nn in n.split('.')
            ]
            new_p_name = '.'.join(blocks)
            full_name = padd_module_name + '_' + new_p_name
            if full_name in w_dict:
                new_state_dict[n] = torch.tensor(w_dict[full_name])
                if 'weight' in n:
                    new_state_dict[n] = new_state_dict[n].T
                print('matched: ', f"{full_name} --> {n}", new_state_dict[n].shape)
            else:
                print('not match: ', full_name)
                import pdb; pdb.set_trace()
        self.load_state_dict(new_state_dict)


class post_process_layer(nn.Module):

    def __init__(self, process_cmd, hidden_size,
                    dropout_rate=0, postfix='') -> None:
        super().__init__()
        self.process_cmd = process_cmd
        self.dropout_rate = dropout_rate
        self.postfix = postfix
        
        for cmd in self.process_cmd:
            if cmd == "n":  # add layer normalization
                self.layer_norm = nn.LayerNorm(hidden_size)
            elif cmd == "d":  # add dropout
                if self.dropout_rate:
                    self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, prev_out, out):
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                out = out + prev_out if prev_out is not None else out
            elif cmd == "n":  # add layer normalization
                out = self.layer_norm(out)
            elif cmd == "d":  # add dropout
                if self.dropout_rate:
                    out = self.dropout(out)
        return out
    
    def load_paddle_weight(self, w_dict, name_prefix):
        padd_module_name = name_prefix + self.postfix
        translate_name = {
            'weight': 'scale',
            'bias': 'bias',
        }

        new_state_dict = {}
        for n, p in self.named_parameters():
            blocks = [
                translate_name[nn] if nn in translate_name else nn
                for nn in n.split('.')
            ]
            new_p_name = '_'.join(blocks)
            full_name = padd_module_name + '_' + new_p_name
            if full_name in w_dict:
                new_state_dict[n] = torch.tensor(w_dict[full_name])
                print('matched: ', f"{full_name} --> {n}", new_state_dict[n].shape)
            else:
                print('not match: ', full_name)
                import pdb; pdb.set_trace()
        self.load_state_dict(new_state_dict)


class pre_process_layer(post_process_layer):
    
    def forward(self, out):
        return super().forward(None, out)


class encoder_co_layer(nn.Module):

    def __init__(self,
                co_head,
                co_key,
                co_value,
                co_model,
                d_model,
                d_inner_hid,
                v_model,
                v_inner_hid,
                prepostprocess_dropout,
                attention_dropout,
                relu_dropout,
                hidden_act,
                preprocess_cmd="n",
                postprocess_cmd="da",
                postfix='') -> None:
        super().__init__()
        self.postfix = postfix
        self.co_head = co_head
        self.co_key = co_key
        self.co_value = co_value
        self.co_model = co_model
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.v_model = v_model
        self.v_inner_hid = v_inner_hid
        self.prepostprocess_dropout = prepostprocess_dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.hidden_act = hidden_act
        self.preprocess_cmd = preprocess_cmd
        self.postprocess_cmd = postprocess_cmd
        self.build_layers()
    
    def load_paddle_weight(self, w_dict, name_prefix):
        padd_module_name = name_prefix + self.postfix

        for ch in self.children():
           ch.load_paddle_weight(w_dict, padd_module_name)            
    
    def build_layers(self):
        self.pre_att = pre_process_layer(
            self.preprocess_cmd,
            self.d_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_pre_att'
        )
        self.vl_pre_att = pre_process_layer(
            self.preprocess_cmd,
            self.co_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_vl_pre_att'
        )
        
        self.multi_head_att = multi_head_attention(
            self.co_key,
            self.co_value,
            self.d_model,
            self.v_model,
            n_head=self.co_head,
            dropout_rate=self.attention_dropout,
            postfix='_multi_head_att'
        )
        self.vl_multi_head_att = multi_head_attention(
            self.co_key,
            self.co_value,
            self.v_model,
            self.d_model,
            n_head=self.co_head,
            dropout_rate=self.attention_dropout,
            postfix='_vl_multi_head_att'
        )
        
        self.post_att = post_process_layer(
            self.postprocess_cmd,
            self.d_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_post_att'
        )
        self.vl_post_att = post_process_layer(
            self.postprocess_cmd,
            self.v_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_vl_post_att'
        )

        self.pre_ffn = pre_process_layer(
            self.preprocess_cmd,
            self.d_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_pre_ffn'
        )
        self.vl_pre_ffn = pre_process_layer(
            self.preprocess_cmd,
            self.v_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_pre_vl_ffn'
        )

        self.ffn = positionwise_feed_forward(
            self.d_inner_hid,
            self.d_model,
            self.relu_dropout,
            hidden_act_fn=self.hidden_act,
            postfix='_ffn',
        )
        self.vl_ffn = positionwise_feed_forward(
            self.v_inner_hid,
            self.v_model,
            self.relu_dropout,
            hidden_act_fn=self.hidden_act,
            postfix='_vl_ffn',
        )

        self.post_ffn = post_process_layer(
            self.postprocess_cmd,
            self.d_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_post_ffn'
        )
        self.vl_post_ffn = post_process_layer(
            self.postprocess_cmd,
            self.v_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_vl_post_ffn'
        )
    
    def forward(self,
                enc_input: torch.Tensor,
                enc_vl_input: torch.Tensor,
                attn_vl_bias: torch.Tensor) -> torch.Tensor:
        enc_input_pre = self.pre_att(enc_input)
        enc_input_vl_pre = self.vl_pre_att(enc_vl_input)

        attn_output = self.multi_head_att(
            enc_input_pre, enc_input_vl_pre, enc_input_vl_pre,
            attn_vl_bias.permute([0, 1, 3, 2])
        )
        attn_vl_output = self.vl_multi_head_att(
            enc_input_vl_pre, enc_input_pre, enc_input_pre,
            attn_vl_bias,
        )

        attn_output = self.post_att(enc_input, attn_output)
        attn_vl_output = self.vl_post_att(enc_vl_input, attn_vl_output)
        
        pre_attn_output = self.pre_ffn(attn_output)
        pre_attn_vl_output = self.vl_pre_ffn(attn_vl_output)
        ffd_output = self.ffn(pre_attn_output)
        ffd_vl_output = self.vl_ffn(pre_attn_vl_output)

        enc_output = self.post_ffn(attn_output, ffd_output)
        enc_vl_output = self.vl_post_ffn(attn_vl_output, ffd_vl_output)

        return enc_output, enc_vl_output


class encoder_layer(nn.Module):

    def __init__(self,
                n_head,
                d_key,
                d_value,
                d_model,
                d_inner_hid,
                prepostprocess_dropout,
                attention_dropout,
                relu_dropout,
                hidden_act,
                preprocess_cmd="n",
                postprocess_cmd="da",
                postfix='') -> None:
        super().__init__()
        self.postfix = postfix
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.prepostprocess_dropout = prepostprocess_dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.hidden_act = hidden_act
        self.preprocess_cmd = preprocess_cmd
        self.postprocess_cmd = postprocess_cmd
        self.build_layers()
    
    def load_paddle_weight(self, w_dict, name_prefix):
        padd_module_name = name_prefix + self.postfix

        for ch in self.children():
           ch.load_paddle_weight(w_dict, padd_module_name)            
    
    def build_layers(self):
        self.pre_att = pre_process_layer(
            self.preprocess_cmd,
            self.d_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_pre_att'
        )
        self.multi_head_att = multi_head_attention(
            self.d_key,
            self.d_value,
            self.d_model,
            self.d_model,
            n_head=self.n_head,
            dropout_rate=self.attention_dropout,
            postfix='_multi_head_att'
        )
        self.post_att = post_process_layer(
            self.postprocess_cmd,
            self.d_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_post_att',
        )

        self.pre_ffn = pre_process_layer(
            self.preprocess_cmd,
            self.d_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_pre_ffn',
        )
        self.ffn = positionwise_feed_forward(
            self.d_inner_hid,
            self.d_model,
            self.relu_dropout,
            hidden_act_fn=self.hidden_act,
            postfix='_ffn'
        )
        self.post_ffn = post_process_layer(
            self.postprocess_cmd,
            self.d_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='_post_ffn'
        )
    
    def forward(self, enc_input: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        pre_attn = self.pre_att(enc_input)
        attn_output = self.multi_head_att(pre_attn, None, None, attn_bias)
        attn_output = self.post_att(enc_input, attn_output)
        
        pre_ffd = self.pre_ffn(attn_output)
        ffd_output = self.ffn(pre_ffd)
        post_ffd = self.post_ffn(attn_output, ffd_output)
        return post_ffd


class encoder(nn.Module):

    def __init__(self,
                n_layer,
                n_head,
                d_key,
                d_value,
                d_model,
                d_inner_hid,
                v_head,
                v_key,
                v_value,
                v_model,
                v_inner_hid,
                co_head,
                co_key,
                co_value,
                co_model,
                co_inner_hid,
                prepostprocess_dropout,
                attention_dropout,
                relu_dropout,
                hidden_act,
                preprocess_cmd="n",
                postprocess_cmd="da",
                param_initializer=None,
                v_biattention_id=[0, 1, 2, 3, 4, 5],
                t_biattention_id=[18, 19, 20, 21, 22, 23],
                name=""):
        super().__init__()
        self.name = name
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.v_head = v_head
        self.v_key = v_key
        self.v_value = v_value
        self.v_model = v_model
        self.v_inner_hid = v_inner_hid
        self.co_head = co_head
        self.co_key = co_key
        self.co_value = co_value
        self.co_model = co_model
        self.co_inner_hid = co_inner_hid
        self.prepostprocess_dropout = prepostprocess_dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.hidden_act = hidden_act
        self.preprocess_cmd = preprocess_cmd
        self.postprocess_cmd = postprocess_cmd
        self.param_initializer = param_initializer
        self.v_biattention_id = v_biattention_id
        self.t_biattention_id = t_biattention_id
        self.build_layers()
    
    def load_paddle_weight(self, w_dict, *args):
        for ch in self.children():
           ch.load_paddle_weight(w_dict, self.name)            
    
    def build_layers(self):
        v_start = 0
        t_start = 0
        block = 0
        v_end = None
        t_end = None

        for v_layer_id, t_layer_id in zip(self.v_biattention_id,
                                          self.t_biattention_id):
            v_end = v_layer_id
            t_end = t_layer_id
            
            for idx in range(t_start, t_end):
                layer = encoder_layer(
                    self.n_head,
                    self.d_key,
                    self.d_value,
                    self.d_model,
                    self.d_inner_hid,
                    self.prepostprocess_dropout,
                    self.attention_dropout,
                    self.relu_dropout,
                    self.hidden_act,
                    preprocess_cmd=self.preprocess_cmd,
                    postprocess_cmd=self.postprocess_cmd,
                    postfix=f'_layer_{idx}'
                )
                setattr(self, f'layer_{idx}', layer)
            
            for idx in range(v_start, v_end):
                layer = encoder_layer(
                    self.v_head,
                    self.v_key,
                    self.v_value,
                    self.v_model,
                    self.v_inner_hid,
                    self.prepostprocess_dropout,
                    self.attention_dropout,
                    self.relu_dropout,
                    self.hidden_act,
                    preprocess_cmd=self.preprocess_cmd,
                    postprocess_cmd=self.postprocess_cmd,
                    postfix=f"_vlayer_{idx}",
                )
                setattr(self, f'vlayer_{idx}', layer)
            
            co_layer = encoder_co_layer(
                self.co_head,
                self.co_key,
                self.co_value,
                self.co_model,
                self.d_model,
                self.d_inner_hid,
                self.v_model,
                self.v_inner_hid,
                self.prepostprocess_dropout,
                self.attention_dropout,
                self.relu_dropout,
                self.hidden_act,
                preprocess_cmd=self.preprocess_cmd,
                postprocess_cmd=self.postprocess_cmd,
                postfix=f"_colayer_{block}",
            )
            setattr(self, f'colayer_{block}', co_layer)
            block += 1
            v_start = v_end
            t_start = t_end
        
        layer = encoder_layer(
            self.n_head,
            self.d_key,
            self.d_value,
            self.d_model,
            self.d_inner_hid,
            self.prepostprocess_dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.hidden_act,
            preprocess_cmd=self.preprocess_cmd,
            postprocess_cmd=self.postprocess_cmd,
            postfix=f'_layer_{t_end}',
        )
        setattr(self, f'layer_{t_end}', layer)

        vlayer = encoder_layer(
            self.v_head,
            self.v_key,
            self.v_value,
            self.v_model,
            self.v_inner_hid,
            self.prepostprocess_dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.hidden_act,
            self.preprocess_cmd,
            self.postprocess_cmd,
            postfix=f'_vlayer_{v_end}',
        )
        setattr(self, f'vlayer_{v_end}', vlayer)

        self.post_encoder = pre_process_layer(
            self.preprocess_cmd,
            self.d_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='post_encoder'
        )
        self.vl_post_encoder = pre_process_layer(
            self.preprocess_cmd,
            self.v_model,
            dropout_rate=self.prepostprocess_dropout,
            postfix='vl_post_encoder'
        )
    
    def forward(self,
                enc_input: torch.Tensor,
                enc_vl_input: torch.Tensor,
                attn_bias: torch.Tensor,
                attn_image_bias: torch.Tensor,
                attn_vl_bias: torch.Tensor) -> torch.Tensor:
        v_start = 0
        t_start = 0
        block = 0
        v_end = None
        t_end = None

        enc_output = None
        enc_vl_output = None

        for v_layer_id, t_layer_id in zip(self.v_biattention_id,
                                          self.t_biattention_id):
            v_end = v_layer_id
            t_end = t_layer_id
            
            for idx in range(t_start, t_end):
                layer = getattr(self, f'layer_{idx}', None)
                enc_output = layer(enc_input, attn_bias)
                enc_input = enc_output
            
            for idx in range(v_start, v_end):
                vlayer = getattr(self, f'vlayer_{idx}', None)
                enc_vl_output = vlayer(enc_vl_input, attn_image_bias)
                enc_vl_input = enc_vl_output
            
            colayer = getattr(self, f'colayer_{block}', None)
            enc_output, enc_vl_output = colayer(enc_input, enc_vl_input, attn_vl_bias)
            enc_input, enc_vl_input = enc_output, enc_vl_output

            block += 1
            v_start = v_end
            t_start = t_end
        
        layer = getattr(self, f'layer_{t_end}', None)
        enc_output = layer(enc_output, attn_bias)
        
        vlayer = getattr(self, f'vlayer_{v_end}', None)
        enc_vl_output = vlayer(enc_vl_output, attn_image_bias)

        enc_output = self.post_encoder(enc_output)
        enc_vl_output = self.vl_post_encoder(enc_vl_output)
        return enc_output, enc_vl_output


def test_multi_head_attention():
    hidden_size = 1024
    seq_len = 80
    img_seq_len = 40
    w_dict = np.load('/home/ron_zhu/Disk2/ernie/export_np.npz')
    
    atten = multi_head_attention(
        64, 64, hidden_size, hidden_size,
        n_head=16, dropout_rate=0.0)
    atten.load_paddle_weight(w_dict, 'encoder_vlayer_3')
    
    np.random.seed(1234)
    A = np.random.normal(size=[2, seq_len, hidden_size])
    A = torch.tensor(A).float()
    B = np.random.normal(size=[2, img_seq_len, hidden_size])
    B = torch.tensor(B).float()
    bias = torch.ones([2, 16, seq_len, img_seq_len])
    out = atten(A, B, B, bias)
    print(out)
    print(out.shape)
    print(out[0][0])


def test_positionwise_feed_forward():
    hidden_size = 1024
    seq_len = 80
    w_dict = np.load('/home/ron_zhu/Disk2/ernie/export_np.npz')
    
    atten = positionwise_feed_forward(
        4096, hidden_size, 0.0, hidden_act_fn=F.gelu)
    atten.load_paddle_weight(w_dict, 'encoder_vlayer_5')
    
    np.random.seed(1234)
    A = np.random.normal(size=[2, seq_len, hidden_size])
    A = torch.tensor(A).float()
    out = atten(A)
    print(out)
    print(out.shape)
    print(out[0][0])


def test_post_process_layer():
    hidden_size = 1024
    seq_len = 80
    w_dict = np.load('/home/ron_zhu/Disk2/ernie/export_np.npz')
    
    atten = post_process_layer(
        "dan", hidden_size,
        dropout_rate=0.0, postfix='_post_att')
    atten.load_paddle_weight(w_dict, 'encoder_layer_9')
    
    np.random.seed(1234)
    A = np.random.normal(size=[2, seq_len, hidden_size])
    A = torch.tensor(A).float()
    B = np.random.normal(size=[2, seq_len, hidden_size])
    B = torch.tensor(B).float()

    out = atten(A, B)
    print(out)
    print(out.shape)
    print(out[0][0])


def test_encoder_co_layer():
    hidden_size = 1024
    seq_len = 80
    img_seq_len = 40
    n_head = 16

    w_dict = np.load('/home/ron_zhu/Disk2/ernie/export_np.npz')
    
    atten = encoder_co_layer(
        n_head, 64, 64, 1024,
        1024, 4096,
        1024, 4096,
        0, 0, 0,
        F.gelu,
        preprocess_cmd="",
        postprocess_cmd="dan",
        postfix="_colayer_2",
    )
    atten.load_paddle_weight(w_dict, 'encoder')
    
    np.random.seed(1234)
    A = np.random.normal(size=[2, seq_len, hidden_size])
    A = torch.tensor(A).float()
    B = np.random.normal(size=[2, img_seq_len, hidden_size])
    B = torch.tensor(B).float()
    Bias = np.random.normal(size=[2, n_head, img_seq_len, seq_len])
    Bias = torch.tensor(Bias).float()

    outputs = atten(A, B, Bias)
    for out in outputs:
        print(out)
        print(out.shape)
        print(out[0][0])

def test_encoder_layer():
    hidden_size = 1024
    seq_len = 80
    img_seq_len = 40
    n_head = 16

    w_dict = np.load('/home/ron_zhu/Disk2/ernie/export_np.npz')
    
    atten = encoder_layer(
        n_head, 64, 64, 1024, 4096,
        0, 0, 0,
        F.gelu,
        preprocess_cmd="",
        postprocess_cmd="dan",
        postfix="_layer_4",
    )
    atten.load_paddle_weight(w_dict, 'encoder')
    
    np.random.seed(1234)
    A = np.random.normal(size=[2, seq_len, hidden_size])
    A = torch.tensor(A).float()
    Bias = np.random.normal(size=[2, n_head, seq_len, seq_len])
    Bias = torch.tensor(Bias).float()

    out = atten(A, Bias)
    print('-' * 100)
    print(out)
    print(out.shape)
    print(out[0][0])


def test_encoder():
    hidden_size = 1024
    seq_len = 80
    img_seq_len = 40
    n_head = 16
    n_layer = 24

    w_dict = np.load('/home/ron_zhu/Disk2/ernie/export_np.npz')
    
    atten = encoder(
        n_layer,
        n_head, 64, 64, 1024, 4096,
        n_head, 64, 64, 1024, 4096,
        n_head, 64, 64, 1024, 4096,
        0, 0, 0,
        F.gelu,
        preprocess_cmd="",
        postprocess_cmd="dan",
        v_biattention_id=[0, 1, 2, 3, 4, 5],
        t_biattention_id=[18, 19, 20, 21, 22, 23],
        name='encoder',
    )
    atten.load_paddle_weight(w_dict)
    
    np.random.seed(1234)
    A = np.random.normal(size=[2, seq_len, hidden_size])
    A = torch.tensor(A).float()
    B = np.random.normal(size=[2, img_seq_len, hidden_size])
    B = torch.tensor(B).float()
    
    aBias = np.random.normal(size=[2, n_head, seq_len, seq_len])
    aBias = torch.tensor(aBias).float()
    bBias = np.random.normal(size=[2, n_head, img_seq_len, img_seq_len])
    bBias = torch.tensor(bBias).float()
    abBias = np.random.normal(size=[2, n_head, img_seq_len, seq_len])
    abBias = torch.tensor(abBias).float()

    atten = atten.cuda()
    A = A.cuda()
    B = B.cuda()
    aBias = aBias.cuda()
    bBias = bBias.cuda()
    abBias = abBias.cuda()

    outputs = atten(A, B, aBias, bBias, abBias)
    for out in outputs:
        print('-' * 100)
        print(out)
        print(out.shape)
        print(out[0][0])


if __name__ == "__main__":
    from loguru import logger
    with logger.catch():
        # test_multi_head_attention()
        # test_encoder_co_layer()
        # test_encoder_layer()
        test_encoder()
