import torch.nn as nn
import torch.utils.data
import torch
import copy
from torch.cuda.amp import autocast as autocast

from torch.nn.init import xavier_normal_ , kaiming_normal_
from functools import partial
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math

from torch.nn.modules.utils import _pair
from .vit_seg_modeling_resnet_skip import ResNetV2

import numpy as np


from os.path import join as pjoin


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"



def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
        
        
class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
      
    """
    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()
        
        #self.shape = shape#H,W
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.num_features = num_features
        #self.batch_size=batch_size
        self.padding=(filter_size-1)//2#in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4*self.num_features, self.filter_size, 1, self.padding)

    
    def forward(self, input, hidden_state):
        hidden,c=hidden_state
        combined = torch.cat((input, hidden), 1)
        A=self.conv(combined)
        (ai,af,ao,ag)=torch.split(A,self.num_features,dim=1)
        i=torch.sigmoid(ai)
        f=torch.sigmoid(af)
        o=torch.sigmoid(ao)
        g=torch.tanh(ag)
        
        next_c=f*c+i*g
        next_h=o*torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self,batch_size,shape):
        return (torch.zeros(batch_size,self.num_features,shape[0],shape[1]).cuda(non_blocking=True) , torch.zeros(batch_size,self.num_features,shape[0],shape[1]).cuda(non_blocking=True))
        # return (torch.zeros(batch_size,self.num_features,shape[0],shape[1]) , torch.zeros(batch_size,self.num_features,shape[0],shape[1]))

def get_weight_init_fn(activation_fn):
    """get weight_initialization function according to activation_fn
    Notes
    -------------------------------------
    if activation_fn requires arguments, use partial() to wrap activation_fn
    """
    fn = activation_fn
    if hasattr( activation_fn , 'func' ):
        fn = activation_fn.func

    if  fn == nn.LeakyReLU:
        negative_slope = 0 
        if hasattr( activation_fn , 'keywords'):
            if activation_fn.keywords.get('negative_slope') is not None:
                negative_slope = activation_fn.keywords['negative_slope']
        if hasattr( activation_fn , 'args'):
            if len( activation_fn.args) > 0 :
                negative_slope = activation_fn.args[0]
        return partial( kaiming_normal_ ,  a = negative_slope )
    elif fn == nn.ReLU or fn == nn.PReLU :
        return partial( kaiming_normal_ , a = 0 )
    else:
        return xavier_normal_
    return

def conv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 , activation_fn= None , use_batchnorm = False , pre_activation = False , bias = True , weight_init_fn = None ):
    """pytorch torch.nn.Conv2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        conv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    conv = nn.Conv2d( in_channels , out_channels , kernel_size , stride , padding , bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    try:
        weight_init_fn( conv.weight )
    except:
        print( conv.weight )
    layers.append( conv )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )

def deconv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 ,  output_padding = 0 , activation_fn = None ,   use_batchnorm = False , pre_activation = False , bias= True , weight_init_fn = None ):
    """pytorch torch.nn.ConvTranspose2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        deconv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))

    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    deconv = nn.ConvTranspose2d( in_channels , out_channels , kernel_size , stride ,  padding , output_padding , bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    weight_init_fn( deconv.weight )
    layers.append( deconv )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )


class BasicBlock(nn.Module):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    use partial() to wrap activation_fn if arguments are needed 
    examples:
        BasicBlock(32,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 , inplace = True ))
    """
    def __init__(self, in_channels , out_channels , kernel_size , stride = 1 , use_batchnorm = False , activation_fn = partial( nn.ReLU ,  inplace=True ) , last_activation_fn = partial( nn.ReLU , inplace=True ) , pre_activation = False , scaling_factor = 1.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv( in_channels , out_channels , kernel_size , stride , kernel_size//2 ,  activation_fn , use_batchnorm )
        self.conv2 = conv( out_channels , out_channels , kernel_size , 1 , kernel_size//2 , None , use_batchnorm  , weight_init_fn = get_weight_init_fn(last_activation_fn) )
        self.downsample = None
        if stride != 1 or in_channels != out_channels :
            self.downsample = conv( in_channels , out_channels , 1 , stride , 0 , None , use_batchnorm )
        if last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        self.scaling_factor = scaling_factor
    def forward(self , x):
        residual = x 
        if self.downsample is not None:
            residual = self.downsample( residual )

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual * self.scaling_factor
        if self.last_activation is not None:
            out = self.last_activation( out )

        return out


def conv5x5_relu(in_channels, out_channels, stride):
    return conv(in_channels, out_channels, 5, stride, 2, activation_fn=partial(nn.ReLU, inplace=True))


def deconv5x5_relu(in_channels, out_channels, stride, output_padding):
    return deconv(in_channels, out_channels, 5, stride, 2, output_padding=output_padding,  activation_fn=partial(nn.ReLU, inplace=True))


def resblock(in_channels):
    """Resblock without BN and the last activation
    """
    return BasicBlock(in_channels, out_channels=in_channels, kernel_size=5, stride=1, use_batchnorm=False, activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=None)


class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(type(self), self).__init__()
        self.conv = conv5x5_relu(in_channels, out_channels, stride)
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        x = self.conv(x)
        x = self.resblock_stack(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.deconv = deconv5x5_relu(
            in_channels, out_channels, stride, output_padding)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.deconv(x)
        return x


class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.conv = conv(in_channels, out_channels, 5, 1, 2, activation_fn=None)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.conv(x)
        return x
    
    


class Attention(nn.Module):
    def __init__(self, vis= False):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(768, self.all_head_size)
        self.key = Linear(768, self.all_head_size)
        self.value = Linear(768, self.all_head_size)

        self.out = Linear(768, 768)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = Linear(768, 3072)
        self.fc2 = Linear(3072, 768)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, in_channels=128):
        super(Embeddings, self).__init__()
        
        # img_size = _pair(img_size)

        # grid_size = (16, 16)
        # patch_size = (img_size[0] // grid_size[0], img_size[1] //  grid_size[1])
        # patch_size_real = (patch_size[0], patch_size[1])
        
        # print(img_size)
        # print(patch_size_real)
        # n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
        n_patches = img_size // 16 * img_size // 16

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=768,
                                       kernel_size=16,
                                       stride=16)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

        self.dropout = Dropout(0.1)


    def forward(self, x):

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # print('x shape {}'.format(x.shape))
        # print('self.position_embeddings shape {}'.format(self.position_embeddings.shape))
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        

        return embeddings


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(768, eps=1e-6)
        for _ in range(12):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        print('input_ids shape:{}'.format(input_ids.shape))
        
        
        embedding_output, features = self.embeddings(input_ids)
        
        print('embedding_output shape:{}'.format(embedding_output.shape))
        print('features:{}'.format(len(features)))
        for f in features:
            print(f.shape)
        # print('')
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        
        print('encoded:{}'.format(encoded.shape))
        return encoded, attn_weights, features


class Net_Trans(nn.Module):
    def __init__(self, level = 3,style = 1, mean_shift = True, range_of_image = 1.0,upsample_fn=partial(torch.nn.functional.interpolate, mode='bilinear')):
        super(type(self), self).__init__()
        self.level = level
        self.style = style
        self.mean_shift = mean_shift
        self.range_of_image = range_of_image
        self.upsample_fn = upsample_fn
        
        self.inblock = EBlock(3 + 3, 32, 1)
        self.eblock1 = EBlock(32, 64, 2)
        self.eblock2 = EBlock(64, 128, 2)
        
        # self.resnet_v2 = ResNetV2(block_units=(3,4,9), width_factor=1)
        
        self.embeddings_1 = Embeddings(img_size=64) # 64 = 256/4
        self.embeddings_2 = Embeddings(img_size=32) # 32 = 128/4
        self.embeddings_3 = Embeddings(img_size=16) # 16 = 64/4
        
        self.encoder = Encoder()
        
        self.dblock1_content = DBlock(256, 64, 2, 1)
        self.dblock2_content = DBlock(128, 32, 2, 1)
        self.outblock_content = OutBlock(64, 3)
        
        self.dblock1_attention = DBlock(256, 64, 2, 1)
        self.dblock2_attention = DBlock(128, 32, 2, 1)
        self.outblock_attention = OutBlock(64, 3)

        
        self.input_padding = None
        
        self.encoder_norm = LayerNorm(768, eps=1e-6)
        self.up_sample_fn = torch.nn.Upsample(scale_factor=4, mode='nearest')
        
        # self.weight_connect = 
        self.weight_connect = nn.Parameter(torch.rand(1), requires_grad=True)
        
        self.conv_more = Conv2dReLU(
            768,
            128,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        print('Net is created.')

    def forward_step(self, input, hidden_state, level):
        # print('hidden state shape:{}'.format(hidden_state.shape))
        
        # x, features = self.resnet_v2(x)
        e32 = self.inblock(input)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        
        # print('e128 shape:{}'.format(e128.shape))
        
        # for f in features:
        #     print(f.shape)
            
        # exit()
        if level == 1:
            embedding_output = self.embeddings_1(e128)
        elif level == 2:
            embedding_output = self.embeddings_2(e128)
        elif level == 3:
            embedding_output = self.embeddings_3(e128)

        # print('after embedding:{}'.format(embedding_output.shape))
        
        encoder_output, attn_weights = self.encoder(self.encoder_norm(
            embedding_output * 0.9 + 0.1* hidden_state))
        
        # print('after encoder:{}'.format(encoder_output.shape))
        
        B, n_patch, hidden = encoder_output.size()
        
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = torch.permute(encoder_output, (0,2,1)).contiguous().view(B, hidden, h, w)
        x = self.upsample_fn(x, scale_factor=16)
        x = self.conv_more(x)
        
        # print(x.shape)
        # print(e128.shape)
        
        d64_content = self.dblock1_content(torch.concat([x, e128], dim=1))
        d32_content = self.dblock2_content(torch.concat([d64_content, e64], dim=1))
        d3_content = self.outblock_content(torch.concat([d32_content, e32], dim=1))
        
        d64_attention = self.dblock1_attention(torch.concat([x, e128], dim=1))
        d32_attention = self.dblock2_attention(torch.concat([d64_attention, e64],dim=1))
        d3_attention = self.outblock_attention(torch.concat([d32_attention, e32],dim=1))
        
        
        d3_content = torch.sigmoid(d3_content)
        d3_attention = torch.tanh(d3_attention)
        
        
        inputs = list(torch.split(input, 3, 1))
        
        # print(inputs[0].shape)
        # print(d3_content.shape)
        # print(d3_attention.shape)
        d3 = d3_content * d3_attention + inputs[0] * (1 - d3_attention)

        return d3, encoder_output, d3_attention

    def forward(self, b1, b2, b3):  
        if self.mean_shift:
            b1 = b1 - 0.5
            b2 = b2 - 0.5
            b3 = b3 - 0.5
            
        b1 = b1 * self.range_of_image
        b2 = b2 * self.range_of_image
        b3 = b3 * self.range_of_image
        bs = b1.shape[0]
        if self.level == 1:
            hidden_state = torch.zeros((bs,16,768), dtype=torch.float32).cuda()
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i2 = torch.zeros(b2.shape).cuda()
            a2 = torch.zeros(b2.shape).cuda()
            
            i1, hidden_state, a1 = self.forward_step(torch.concat((b1,b1), dim=1),hidden_state,1)
        elif self.level == 2:
            hidden_state = torch.zeros((bs,4,768), dtype=torch.float32).cuda()
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i2, hidden_state, a2 = self.forward_step(torch.concat((b2,b2), dim=1),hidden_state,2)
            hidden_state = self.up_sample_fn(hidden_state.transpose(1,2)).transpose(1,2)
            
            i1, hidden_state, a1 = self.forward_step(torch.concat((b1,self.upsample_fn(i2, scale_factor=2)), dim=1),hidden_state,1)
        elif self.level == 3:
            hidden_state = torch.zeros((bs,1,768), dtype=torch.float32).cuda()
        
            i3, hidden_state, a3 = self.forward_step(torch.concat((b3,b3), dim=1),hidden_state,3)
            hidden_state = self.up_sample_fn(hidden_state.transpose(1,2)).transpose(1,2)
            
            i2, hidden_state, a2 = self.forward_step(torch.concat((b2,self.upsample_fn(i3, scale_factor=2)), dim=1),hidden_state,2)
            hidden_state = self.up_sample_fn(hidden_state.transpose(1,2)).transpose(1,2)
            
            i1, hidden_state, a1 = self.forward_step(torch.concat((b1,self.upsample_fn(i2, scale_factor=2)), dim=1),hidden_state,1)
        else:
            raise NotImplementedError('{} is not supported'.format(self.level))
    

        
        i1 = i1 / self.range_of_image
        i2 = i2 / self.range_of_image
        i3 = i3 / self.range_of_image
        
        if self.mean_shift:
            i1 = i1 + 0.5
            i2 = i2 + 0.5
            i3 = i3 + 0.5
            
        return i1, i2, i3, a1, a2, a3
    
