import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np


class TransformerEdgePolicy(nn.Module):
    """docstring for TransformerEdgePolicy."""

    def __init__(self, args):
        super(TransformerEdgePolicy, self).__init__()
        self.args = args
        self.device = args['device']
        self.input_dim = args['input_dim']
        self.edge_dim = args['edge_dim']
        self.sep_self_edge = args['sep_self_edge']
        self.n_hops = args['n_hops']
        self.with_sigma = args['with_sigma']
        self.sep_key_val = args['sep_key_val']
        self.hard_attention = args['hard_attn']
        self.hard_attn_deg = args['hard_attn_deg']
        self.rand_hard_attention = args['rand_hard_attn']
        self.train_attention = args['train_attention']
        self.only_obs_edge = args['only_obs_edge']

        self.model_dim = args['transformer_model_dim']
        self.num_out = 2
        if 'num_out' in args:
            self.num_out = args['num_out']
        self.attn_dts = []
        self.attn_progs = []

        self.use_internal_state = False
        if 'use_internal_state' in args:
            self.use_internal_state = args['use_internal_state']

        self.use_sigmoid = False
        if 'use_sigmoid' in args:
            self.use_sigmoid = args['use_sigmoid']

        self.use_soft_with_prog = False

        self.na_params = []  # all parameters except attention parameters

        assert(self.train_attention or self.sep_key_val) # both cannot be false
        assert(not (self.hard_attention and self.rand_hard_attention)) # both cannot be true

        assert(self.n_hops == 1 or self.n_hops == 2 or self.n_hops == 3)

        if self.use_internal_state:
            self.internal_state = None

        self.fc_edge_k = nn.Linear(self.edge_dim, args['transformer_model_dim'])

        if not self.train_attention:
            for param in self.fc_edge_k.parameters():
                param.requires_grad = False

        self.fc_edge_v = nn.Linear(self.edge_dim, args['transformer_model_dim'])
        self.na_params.extend(self.fc_edge_v.parameters())

        if self.sep_self_edge:
            self.self_edge_k = nn.Parameter(torch.randn(args['transformer_model_dim']))

            if not self.train_attention:
                for param in self.self_edge_k.parameters():
                    param.requires_grad = False

            self.self_edge_v = nn.Parameter(torch.randn(args['transformer_model_dim']))
            self.na_params.extend(self.self_edge_v.parameters())

        self.fc1 = nn.Linear(self.input_dim, args['transformer_model_dim'])

        if not self.train_attention:
            for param in self.fc1.parameters():
                param.requires_grad = False

        if self.sep_key_val:
            self.fc1_v = nn.Linear(self.input_dim, args['transformer_model_dim'])
            self.na_params.extend(self.fc1_v.parameters())

        self.encoder_layer = RelativePosTransformerLayer(args['transformer_model_dim'], args['nhead'], dim_feedforward=args['transformer_fc_dim'], hard_attention = self.hard_attention, hard_attn_deg=self.hard_attn_deg, rand_hard_attention = self.rand_hard_attention, train_attention = self.train_attention, only_obs_edge=self.only_obs_edge, device = self.device)



        self.na_params.extend(self.encoder_layer.na_params)

        if self.n_hops >= 2:
            self.encoder_layer_2 = RelativePosTransformerLayer(args['transformer_model_dim'], args['nhead'], dim_feedforward=args['transformer_fc_dim'], hard_attention = self.hard_attention, rand_hard_attention = self.rand_hard_attention, hard_attn_deg=self.hard_attn_deg,  device = self.device)

            self.na_params.extend(self.encoder_layer_2.na_params)

        if self.n_hops >= 3:
            self.encoder_layer_3 = RelativePosTransformerLayer(args['transformer_model_dim'], args['nhead'], dim_feedforward=args['transformer_fc_dim'], hard_attention = self.hard_attention, rand_hard_attention = self.rand_hard_attention, hard_attn_deg=self.hard_attn_deg,  device = self.device)
            self.na_params.extend(self.encoder_layer_3.na_params)

        self.fc2 = nn.Linear(args['transformer_model_dim'], self.num_out)
        self.na_params.extend(self.fc2.parameters())
        if self.with_sigma:
            self.fc2_sigma = nn.Linear(args['transformer_model_dim'], self.num_out)
            self.na_params.extend(self.fc2_sigma.parameters())
            self.sigma_plus = nn.Softplus()

        if self.use_internal_state:
            self.fc_internal_state = nn.Linear(args['transformer_model_dim'], args['transformer_model_dim'])
            self.na_params.extend(self.fc_internal_state.parameters())

        self.sigmoid  = nn.Sigmoid()
    def use_dt_for_attn(self, dts, use_soft_with_prog=False):
        self.attn_dts = dts
        self.use_soft_with_prog = use_soft_with_prog


    def use_prog_for_attn(self, progs, use_soft_with_prog=False):
        self.attn_progs = progs
        self.use_soft_with_prog = use_soft_with_prog

    def use_prog_for_attn1(self, progs, hop, use_soft_with_prog=False):
        self.attn_progs = progs
        self.use_soft_with_prog = use_soft_with_prog
        self.attn_prog_hop = hop 

    def reset_internal_state(self):
        if self.use_internal_state:
            self.internal_state = None


    def forward(self, input, edge_input, comm_weights, prog_input = None, need_weights=False):
        # Input (N, d_in)
        # edge_input (N^2, d_in_edge)
        input = input.to(self.device)
        x_kq = F.relu(self.fc1(input)) # (N, d_model)
        x_kq = torch.unsqueeze(x_kq, 1) # (N, 1, d_model)
        if self.sep_key_val:
            x_v = F.relu(self.fc1_v(input)) # (N, d_model)
            x_v = torch.unsqueeze(x_v, 1) # (N, 1, d_model)
        else:
            x_v = x_kq # (N, 1, d_model)

        if self.use_internal_state and self.internal_state != None:
            x_v = x_v + self.internal_state.unsqueeze(1)


        edge_input = edge_input.to(self.device)
        edge_k = F.relu(self.fc_edge_k(edge_input)) # (N^2, d_model)
        edge_v = F.relu(self.fc_edge_v(edge_input)) # (N^2, d_model)

        if self.sep_self_edge:
            N = input.size()[0]
            diag_index = torch.eye(N, dtype=torch.bool, device=self.device)
            edge_k = edge_k.contiguous().view(N, N, -1)
            edge_k[diag_index] = self.self_edge_k
            edge_k = edge_k.contiguous().view(N*N, -1)

            edge_v = edge_v.contiguous().view(N, N, -1)
            edge_v[diag_index] = self.self_edge_v
            edge_v = edge_v.contiguous().view(N*N, -1)

        edge_k = torch.unsqueeze(edge_k, 1) # (N^2, 1, d_model)
        edge_v = torch.unsqueeze(edge_v, 1) # (N^2, 1, d_model)

        if len(self.attn_dts) > 0:
            assert(len(self.attn_dts) == self.n_hops)
            S, N, _, d = prog_input.shape

            dt_att_weights = []
            for dt in self.attn_dts:
                pinput = prog_input.reshape(S*N*N,-1).detach().cpu().numpy()
                attn = torch.tensor(dt.predict(pinput), dtype=torch.float32).to(self.device)
                dt_att_weights.append(attn.view(S,N,N))
        elif len(self.attn_progs) > 0:
            if len(self.attn_progs) == self.n_hops:
                dt_att_weights = []
                for prog in self.attn_progs:
                    dt_att_weights.append(prog.eval(prog_input))
            else:
                dt_att_weights = [None for _ in range(self.n_hops)]
                dt_att_weights[self.attn_prog_hop] = self.attn_progs[0].eval(prog_input)
            self.prog_attn_wghts = dt_att_weights[0].clone().detach()
        else:
            dt_att_weights = [None for _ in range(self.n_hops)]



        x, att_weights1 = self.encoder_layer(x_kq, x_v, edge_k, edge_v, comm_weights, dt_att_weights[0], soft_with_prog=self.use_soft_with_prog) # (N, 1, d_model)
        if self.n_hops >= 2:
            assert(not self.sep_key_val) # not yet supported with different k and v encoding
            x, att_weights2 = self.encoder_layer_2(x_kq, x, edge_k, edge_v, comm_weights, dt_att_weights[1], soft_with_prog=self.use_soft_with_prog) # (N, 1, d_model)

        if self.n_hops >= 3:
            assert(not self.sep_key_val) # not yet supported with different k and v encoding
            x, att_weights3 = self.encoder_layer_3(x_kq, x, edge_k, edge_v, comm_weights, dt_att_weights[2], soft_with_prog=self.use_soft_with_prog) # (N, 1, d_model)


        x = torch.squeeze(x) # (N, d_model)
        action = self.fc2(x)
        if self.use_sigmoid:
            action = self.sigmoid(action)
        if self.with_sigma:
            if self.use_sigmoid:
                sigma = self.sigmoid(self.fc2_sigma(x))
            else:
                sigma = self.sigma_plus(self.fc2_sigma(x))


        if self.use_internal_state:
            self.internal_state = self.fc_internal_state(x)

        if need_weights:
            if self.n_hops == 1:
                if self.with_sigma:
                    return action, sigma, [att_weights1]
                else:
                    return action, [att_weights1] # (1, h, N, N)
            elif self.n_hops == 2:
                if self.with_sigma:
                    return action, sigma, [att_weights1, att_weights2]
                else:
                    return action, [att_weights1, att_weights2]
            elif self.n_hops == 3:
                if self.with_sigma:
                    return action, sigma, [att_weights1, att_weights2, att_weights3]
                else:
                    return action, [att_weights1, att_weights2, att_weights3]
        else:
            if self.with_sigma:
                return action, sigma
            return action


class RelativePosTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", hard_attention=False, hard_attn_deg=2, rand_hard_attention = False, train_attention = True, only_obs_edge=False, device = None):
        super(RelativePosTransformerLayer, self).__init__()
        self.device = device
        self.train_attention = train_attention

        self.na_params = []

        self.self_attn = EdgeMultiheadAttention(d_model, nhead, dropout=dropout, hard_attention = hard_attention, hard_attn_deg=hard_attn_deg, rand_hard_attention = rand_hard_attention, only_obs_edge=only_obs_edge)

        if not self.train_attention:
            for param in self.self_attn.parameters():
                param.requires_grad = False

        self.value_comp = EdgeMultiheadValue(d_model, nhead, dropout=dropout)
        self.na_params.extend(self.value_comp.parameters())

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.na_params.extend(self.linear1.parameters())
        self.dropout = nn.Dropout(dropout)
        self.na_params.extend(self.dropout.parameters())
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.na_params.extend(self.linear2.parameters())

        self.norm1 = nn.LayerNorm(d_model)
        self.na_params.extend(self.norm1.parameters())
        self.norm2 = nn.LayerNorm(d_model)
        self.na_params.extend(self.norm2.parameters())
        self.dropout1 = nn.Dropout(dropout)
        self.na_params.extend(self.dropout1.parameters())
        self.dropout2 = nn.Dropout(dropout)
        self.na_params.extend(self.dropout2.parameters())

        self.activation = _get_activation_fn(activation)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(RelativePosTransformerLayer, self).__setstate__(state)

    def forward(self, src_kq, src_v, edge_k, edge_v, comm_weights, dt_att_weights, need_weights=True, soft_with_prog=False):
        """
        src: (N, 1, d_model)
        edge_k, edge_v: (N^2, 1, d_model)
            First N entries are relative pos to first agent, etc.
        """
        if dt_att_weights == None:
            # compute att weights with transformer
            N, bsz, _ = src_kq.size()

            mask = torch.randint(100, (bsz, N, N)).to(self.device) >= comm_weights
            #print(mask)
            mask = mask.to(self.device)
            #assert(bsz == 1)

            att_weights = self.self_attn(src_kq, src_kq, edge_k, key_padding_mask = mask)

        elif soft_with_prog:
            # compute att weights with transformer
            N, bsz, _ = src_kq.size()

            mask = torch.randint(100, (bsz, N, N)).to(self.device) >= comm_weights
            mask = mask.to(self.device)
            assert(bsz == 1)

            # Use prog attention weights to restrict comms
            mask[dt_att_weights<1e-2] = True

            att_weights = self.self_attn(src_kq, src_kq, edge_k, key_padding_mask = mask)

            att_weights.masked_fill_(mask, 0)

        else:
            att_weights = dt_att_weights

        src2 = self.value_comp(att_weights, src_v, edge_v)

        src = src_v + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if need_weights:
            return src, att_weights
        else:
            return src, None

class EdgeMultiheadAttention(nn.Module):
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, hard_attention = False, hard_attn_deg=2, rand_hard_attention = False, only_obs_edge=False, dropout=0., bias=True):
        super(EdgeMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.hard_attention = hard_attention
        self.hard_attn_deg=hard_attn_deg
        self.rand_hard_attention = rand_hard_attention
        self.only_obs_edge = only_obs_edge
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(2 * embed_dim, embed_dim))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        #self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(2 * embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(EdgeMultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, edge_k, key_padding_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        """
        Shape:
        - Inputs:
        - query: :math:`(S, N, E)` where S is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
          the embedding dimension.
        -  key_padding_mask: :math:`(N, S, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - edge_k, edge_v: (S*S, N, E)

        - Outputs:
        - attn_output_weights: :math:`(N, S, S)` where N is the batch size,
          S is the target sequence length, S is the source sequence length.
        """

        seq_len, bsz, embed_dim = query.size()

        head_dim = self.head_dim
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        dropout_p = self.dropout

        scaling = float(head_dim) ** -0.5

        if torch.equal(query, key):
            # self-attention
            q, k = F.linear(query, in_proj_weight, in_proj_bias).chunk(2, dim=-1)

        else:
            assert(False)

        # q, k: (S, N, E)
        q = q * scaling

        q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # q, k: (N*h, S, E/h)

        src_len = k.size(1)
        assert(src_len == seq_len)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        # Add contribution from edge, attention weights
        # edge k: (S^2, N, E)
        edge_k = edge_k.contiguous().view(seq_len*seq_len, bsz * num_heads, head_dim).transpose(0, 1)

        # edge k: (N*h, S^2, E/h)
        edge_k = edge_k.contiguous().view(bsz*num_heads*seq_len, seq_len, head_dim).transpose(1, 2) # (N*h*S, E/h, S)
        q_bar = q.contiguous().view(bsz*num_heads*seq_len, 1, head_dim) # (N*h*S, 1, E/h)
        att_weight_edge = torch.bmm(q_bar, edge_k).contiguous().view(bsz*num_heads, seq_len, seq_len) #(N*h, S, S)

        if self.only_obs_edge:
            attn_output_weights = att_weight_edge
        else:
            # Attention weights
            attn_output_weights = torch.bmm(q, k.transpose(1, 2)) # (N*h, S, S)
            assert list(attn_output_weights.size()) == [bsz * num_heads, seq_len, seq_len]
            attn_output_weights = attn_output_weights + att_weight_edge

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, seq_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1),

            #float('-inf'),
            math.log(1e-45),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, seq_len, src_len)
            #print(attn_output_weights)



        if self.rand_hard_attention:
            num_select = 2
            attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)
            S, N, _ = attn_output_weights.shape
            attn_output_weights = attn_output_weights.view(S*N, N)
            idx = torch.multinomial(attn_output_weights, num_select)
            attn_output_weights.fill_(float('-inf'))

            attn_output_weights[np.arange(S*N).repeat(num_select), idx.view(-1)] = 1.0

            attn_output_weights = attn_output_weights.view(S, N, N)

        if self.hard_attention:
            num_agents_to_ignore = math.floor(seq_len - self.hard_attn_deg)
            _, smallest = torch.topk(attn_output_weights, num_agents_to_ignore, dim=2, largest=False) # (N*h, S, S*(1 - self.ratio_to_attend))
            attn_output_weights.scatter_(2, smallest, float('-inf'))
            
        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)


        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=self.training)


        return attn_output_weights



class EdgeMultiheadValue(nn.Module):
    __constants__ = ['v_proj_weight', 'in_proj_weight_v']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(EdgeMultiheadValue, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight_v = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight_v)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(EdgeMultiheadValue, self).__setstate__(state)

    def forward(self, attn_output_weights, value, edge_v):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        """
        Shape:
        - Inputs:
        - attn_output_weights: :math:`(N*h, S, S)` where S is the target sequence length, N is the batch size, h is
          the number of heads.
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        -  edge_v: (S*S, N, E)

        - Outputs:
        - attn_output: :math:`(S, N, E)` where S is the target sequence length, N is the batch size,
          E is the embedding dimension.
        """

        seq_len, bsz, embed_dim = value.size()

        head_dim = self.head_dim
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight_v
        in_proj_bias = self.in_proj_bias
        dropout_p = self.dropout


        v = F.linear(value, in_proj_weight, in_proj_bias)

        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        #v: (N*h, S, E/h)

        # Attention output
        attn_output = torch.bmm(attn_output_weights, v)

        assert list(attn_output.size()) == [bsz * num_heads, seq_len, head_dim]

        # Add contribution from edge, value
        edge_v = edge_v.contiguous().view(bsz*num_heads*seq_len, seq_len, head_dim) # (N*h*S, S, E/h)
        attn_output_weights_bar = attn_output_weights.contiguous().view(bsz*num_heads*seq_len, 1, seq_len) # (N*h*S, 1, S)
        att_output_edge = torch.bmm(attn_output_weights_bar, edge_v).contiguous().view(bsz*num_heads, seq_len, head_dim) # (N*h, S, E/h)
        attn_output = attn_output + att_output_edge

        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim) # (S, N, E)


        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias) # (S, N, E)

        return attn_output


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


