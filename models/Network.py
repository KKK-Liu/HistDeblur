import torch.nn as nn
import torch.utils.data
import torch

from torch.cuda.amp import autocast as autocast

from torch.nn.init import xavier_normal_ , kaiming_normal_
from functools import partial

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


class Net(nn.Module):
    def __init__(self,level = 3,style = 1, mean_shift = True, range_of_image = 1.0,upsample_fn=partial(torch.nn.functional.interpolate, mode='bilinear'), xavier_init_all=True):
        super(type(self), self).__init__()
        self.level = level
        self.style = style
        self.mean_shift = mean_shift
        self.range_of_image = range_of_image
        self.upsample_fn = upsample_fn
        self.inblock = EBlock(3 + 3, 32, 1)
        self.eblock1 = EBlock(32, 64, 2)
        self.eblock2 = EBlock(64, 128, 2)
        
        self.convlstm = CLSTM_cell(128, 128, 5)
        
        self.dblock1_content = DBlock(128, 64, 2, 1)
        self.dblock2_content = DBlock(64, 32, 2, 1)
        self.outblock_content = OutBlock(32, 3)
        
        self.dblock1_attention = DBlock(128, 64, 2, 1)
        self.dblock2_attention = DBlock(64, 32, 2, 1)
        self.outblock_attention = OutBlock(32, 3)

        self.mask_weight = torch.nn.Parameter(torch.zeros((1),dtype=torch.float32,requires_grad=True))
        
        self.input_padding = None
        # if xavier_init_all:
        #     for name, m in self.named_modules():
        #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #             torch.nn.init.xavier_normal_(m.weight)
        #             # torch.nn.init.kaiming_normal_(m.weight)
        #             print(name)
        
        print('Net is created.')

    def forward_step(self, x, hidden_state):
        
        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        h, c = self.convlstm(e128, hidden_state)
        
        d64_content = self.dblock1_content(h)
        d32_content = self.dblock2_content(d64_content + e64)
        d3_content = self.outblock_content(d32_content + e32)
        
        d64_attention = self.dblock1_attention(h)
        d32_attention = self.dblock2_attention(d64_attention + e64)
        d3_attention = self.outblock_attention(d32_attention + e32)
        
        
        # d3_attention = torch.nn.functional.softmax(d3_attention, dim=1)
        d3_content = torch.sigmoid(d3_content)
        d3_attention = torch.tanh(d3_attention)
        
        # d3_content = torch.tanh(d3_content)
        # d3_attention = torch.sigmoid(d3_attention)
        
        xs = list(torch.split(x, 3, 1))
        if self.style == 1:
            d3 = d3_content * d3_attention * self.mask_weight+ xs[0] *(1-self.mask_weight)
        elif self.style == 2:
            d3 = d3_content * d3_attention + xs[0] * (1 - d3_attention)
        elif self.style == 3:
            d3 = d3_content * d3_attention + xs[0]
        elif self.style == 4:
            d3 = d3_content

        return d3, d3_attention,h,c

    def forward(self, b1, b2, b3, b4):
        if self.mean_shift:
            b1 = b1 - 0.5
            b2 = b2 - 0.5
            b3 = b3 - 0.5
            b4 = b4 - 0.5
            
        b1 = b1 * self.range_of_image
        b2 = b2 * self.range_of_image
        b3 = b3 * self.range_of_image
        b4 = b4 * self.range_of_image
        
        if self.level == 1:
            h, c = self.convlstm.init_hidden(b1.shape[0], (b1.shape[-2]//4, b1.shape[-1]//4))

            i1, a1,h,c = self.forward_step(
                torch.cat([b1, torch.zeros_like(b1)], 1), (h, c))

            i2 = torch.zeros(b2.shape).cuda()
            a2 = torch.zeros(b2.shape).cuda()
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
        
        if self.level == 2:
            h, c = self.convlstm.init_hidden(b2.shape[0], (b2.shape[-2]//4, b2.shape[-1]//4))

            i2, a2,h,c = self.forward_step(
                torch.cat([b2, torch.zeros_like(b2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
            
        if self.level == 3:
            h, c = self.convlstm.init_hidden(b3.shape[0], (b3.shape[-2]//4, b3.shape[-1]//4))

            i3, a3,h,c = self.forward_step(
                torch.cat([b3, torch.zeros_like(b3)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i2, a2,h,c = self.forward_step(
                torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
            
        if self.level == 4:
            h, c = self.convlstm.init_hidden(b4.shape[0], (b4.shape[-2]//4, b4.shape[-1]//4))
        
            i4, a4,h,c = self.forward_step(
                torch.cat([b4, torch.zeros_like(b4)], 1), (h, c))
            
            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            
            i3, a3,h,c = self.forward_step(
                torch.cat([b3, self.upsample_fn(i4, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i2, a2,h,c = self.forward_step(
                torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
        
        i1 = i1 / self.range_of_image
        i2 = i2 / self.range_of_image
        i3 = i3 / self.range_of_image
        i4 = i4 / self.range_of_image
        
        if self.mean_shift:
            i1 = i1 + 0.5
            i2 = i2 + 0.5
            i3 = i3 + 0.5
            i4 = i4 + 0.5
            
        return i1, i2, i3, i4, a1, a2, a3, a4
    
    

class Net_noLSTM(nn.Module):
    def __init__(self,level = 3,style = 1, mean_shift = True, range_of_image = 1.0, nblocks = 3,upsample_fn=partial(torch.nn.functional.interpolate, mode='bilinear'), xavier_init_all=True):
        super(type(self), self).__init__()
        self.level = level
        self.style = style
        self.mean_shift = mean_shift
        self.range_of_image = range_of_image
        self.upsample_fn = upsample_fn
        self.inblock = EBlock(3 + 3, 32, 1)
        self.eblock1 = EBlock(32, 64, 2)
        self.eblock2 = EBlock(64, 128, 2)
        
        # self.convlstm = CLSTM_cell(128, 128, 5)
        
        self.resBlocks = nn.Sequential(*[resblock(128) for i in range(nblocks)])
        
        self.dblock1_content = DBlock(128, 64, 2, 1)
        self.dblock2_content = DBlock(64, 32, 2, 1)
        self.outblock_content = OutBlock(32, 3)
        
        self.dblock1_attention = DBlock(128, 64, 2, 1)
        self.dblock2_attention = DBlock(64, 32, 2, 1)
        self.outblock_attention = OutBlock(32, 3)

        self.mask_weight = torch.nn.Parameter(torch.zeros((1),dtype=torch.float32,requires_grad=True))
        
        self.input_padding = None
        # if xavier_init_all:
        #     for name, m in self.named_modules():
        #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #             torch.nn.init.xavier_normal_(m.weight)
        #             # torch.nn.init.kaiming_normal_(m.weight)
        #             print(name)
        print('Net no lstm is created.')

    def forward_step(self, x):
        
        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        
        d128 = self.resBlocks(e128)
        
        d64_content = self.dblock1_content(d128)
        d32_content = self.dblock2_content(d64_content + e64)
        d3_content = self.outblock_content(d32_content + e32)
        
        d64_attention = self.dblock1_attention(d128)
        d32_attention = self.dblock2_attention(d64_attention + e64)
        d3_attention = self.outblock_attention(d32_attention + e32)
        
        
        # d3_attention = torch.nn.functional.softmax(d3_attention, dim=1)
        
        # d3_content = torch.sigmoid(d3_content)
        # d3_attention = torch.tanh(d3_attention)
        
        d3_content = torch.tanh(d3_content) / 2
        d3_attention = torch.sigmoid(d3_attention)
        
        xs = list(torch.split(x, 3, 1))
        
        if self.style == 1:
            d3 = d3_content * d3_attention * self.mask_weight+ xs[0] *(1-self.mask_weight)
        elif self.style == 2:
            d3 = d3_content * d3_attention + xs[0] * (1 - d3_attention)
        elif self.style == 3:
            d3 = d3_content * d3_attention + xs[0]
        elif self.style == 4:
            d3 = d3_content

        return d3, d3_attention

    def forward(self, b1, b2, b3, b4):
        if self.mean_shift:
            b1 = b1 - 0.5
            b2 = b2 - 0.5
            b3 = b3 - 0.5
            b4 = b4 - 0.5
            
        b1 = b1 * self.range_of_image
        b2 = b2 * self.range_of_image
        b3 = b3 * self.range_of_image
        b4 = b4 * self.range_of_image
        
        if self.level == 1:
            i1, a1 = self.forward_step(torch.cat([b1, b1], 1))

            i2 = torch.zeros(b2.shape).cuda()
            a2 = torch.zeros(b2.shape).cuda()
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
        
        if self.level == 2:
            i2, a2 = self.forward_step(torch.cat([b2, b2], 1))
            i1, a1 = self.forward_step(torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1))
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
            
        if self.level == 3:
            i3, a3 = self.forward_step(torch.cat([b3, b3], 1))
            i2, a2 = self.forward_step(torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1))
            i1, a1 = self.forward_step(torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1))
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
            
        if self.level == 4:
            i4, a4 = self.forward_step(torch.cat([b4, b4], 1))
            i3, a3 = self.forward_step(torch.cat([b3, self.upsample_fn(i4, scale_factor=2)], 1))
            i2, a2 = self.forward_step(torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1))
            i1, a1 = self.forward_step(torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1))
        
        i1 = i1 / self.range_of_image
        i2 = i2 / self.range_of_image
        i3 = i3 / self.range_of_image
        i4 = i4 / self.range_of_image
        
        if self.mean_shift:
            i1 = i1 + 0.5
            i2 = i2 + 0.5
            i3 = i3 + 0.5
            i4 = i4 + 0.5
            
        return i1, i2, i3, i4, a1, a2, a3, a4




class Net_concat(nn.Module):
    def __init__(self,level = 3,style = 1, mean_shift = True, range_of_image = 1.0,upsample_fn=partial(torch.nn.functional.interpolate, mode='bilinear'), xavier_init_all=True):
        super(type(self), self).__init__()
        self.level = level
        self.style = style
        self.mean_shift = mean_shift
        self.range_of_image = range_of_image
        self.upsample_fn = upsample_fn
        self.inblock = EBlock(3 + 3, 32, 1)
        self.eblock1 = EBlock(32, 64, 2)
        self.eblock2 = EBlock(64, 128, 2)
        
        self.convlstm = CLSTM_cell(128, 128, 5)
        
        self.dblock1_content = DBlock(128, 64, 2, 1)
        self.dblock2_content = DBlock(128, 32, 2, 1)
        self.outblock_content = OutBlock(64, 3)
        
        self.dblock1_attention = DBlock(128, 64, 2, 1)
        self.dblock2_attention = DBlock(128, 32, 2, 1)
        self.outblock_attention = OutBlock(64, 3)

        self.mask_weight = torch.nn.Parameter(torch.zeros((1),dtype=torch.float32,requires_grad=True))
        
        self.input_padding = None
        # if xavier_init_all:
        #     for name, m in self.named_modules():
        #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #             torch.nn.init.xavier_normal_(m.weight)
        #             # torch.nn.init.kaiming_normal_(m.weight)
        #             print(name)
        
        print('Net is created.')

    def forward_step(self, x, hidden_state):
        
        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        h, c = self.convlstm(e128, hidden_state)
        
        d64_content = self.dblock1_content(h)
        d32_content = self.dblock2_content(torch.concat([d64_content, e64], dim=1))
        d3_content = self.outblock_content(torch.concat([d32_content, e32], dim=1))
        
        d64_attention = self.dblock1_attention(h)
        d32_attention = self.dblock2_attention(torch.concat([d64_attention, e64],dim=1))
        d3_attention = self.outblock_attention(torch.concat([d32_attention, e32],dim=1))
        
        
        # d3_attention = torch.nn.functional.softmax(d3_attention, dim=1)
        d3_content = torch.sigmoid(d3_content)
        d3_attention = torch.tanh(d3_attention)
        
        # d3_content = torch.tanh(d3_content)
        # d3_attention = torch.sigmoid(d3_attention)
        
        xs = list(torch.split(x, 3, 1))
        if self.style == 1:
            d3 = d3_content * d3_attention * self.mask_weight+ xs[0] *(1-self.mask_weight)
        elif self.style == 2:
            d3 = d3_content * d3_attention + xs[0] * (1 - d3_attention)
        elif self.style == 3:
            d3 = d3_content * d3_attention + xs[0]
        elif self.style == 4:
            d3 = d3_content

        return d3, d3_attention,h,c

    def forward(self, b1, b2, b3, b4):
        if self.mean_shift:
            b1 = b1 - 0.5
            b2 = b2 - 0.5
            b3 = b3 - 0.5
            b4 = b4 - 0.5
            
        b1 = b1 * self.range_of_image
        b2 = b2 * self.range_of_image
        b3 = b3 * self.range_of_image
        b4 = b4 * self.range_of_image
        
        if self.level == 1:
            h, c = self.convlstm.init_hidden(b1.shape[0], (b1.shape[-2]//4, b1.shape[-1]//4))

            i1, a1,h,c = self.forward_step(
                torch.cat([b1, torch.zeros_like(b1)], 1), (h, c))

            i2 = torch.zeros(b2.shape).cuda()
            a2 = torch.zeros(b2.shape).cuda()
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
        
        if self.level == 2:
            h, c = self.convlstm.init_hidden(b2.shape[0], (b2.shape[-2]//4, b2.shape[-1]//4))

            i2, a2,h,c = self.forward_step(
                torch.cat([b2, torch.zeros_like(b2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
            
        if self.level == 3:
            h, c = self.convlstm.init_hidden(b3.shape[0], (b3.shape[-2]//4, b3.shape[-1]//4))

            i3, a3,h,c = self.forward_step(
                torch.cat([b3, torch.zeros_like(b3)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i2, a2,h,c = self.forward_step(
                torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
            
        if self.level == 4:
            h, c = self.convlstm.init_hidden(b4.shape[0], (b4.shape[-2]//4, b4.shape[-1]//4))
        
            i4, a4,h,c = self.forward_step(
                torch.cat([b4, torch.zeros_like(b4)], 1), (h, c))
            
            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            
            i3, a3,h,c = self.forward_step(
                torch.cat([b3, self.upsample_fn(i4, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i2, a2,h,c = self.forward_step(
                torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
        
        i1 = i1 / self.range_of_image
        i2 = i2 / self.range_of_image
        i3 = i3 / self.range_of_image
        i4 = i4 / self.range_of_image
        
        if self.mean_shift:
            i1 = i1 + 0.5
            i2 = i2 + 0.5
            i3 = i3 + 0.5
            i4 = i4 + 0.5
            
        return i1, i2, i3, i4, a1, a2, a3, a4
    
class Net_concat_small(nn.Module):
    def __init__(self,level = 3,style = 1, mean_shift = True, range_of_image = 1.0,upsample_fn=partial(torch.nn.functional.interpolate, mode='bilinear'), xavier_init_all=True):
        super(type(self), self).__init__()
        self.level = level
        self.style = style
        self.mean_shift = mean_shift
        self.range_of_image = range_of_image
        self.upsample_fn = upsample_fn
        self.inblock = EBlock(3 + 3, 16, 1)
        self.eblock1 = EBlock(16, 32, 2)
        self.eblock2 = EBlock(32, 64, 2)
        
        self.convlstm = CLSTM_cell(64, 64, 5)
        
        self.dblock1_content = DBlock(128, 32, 2, 1)
        self.dblock2_content = DBlock(64, 16, 2, 1)
        self.outblock_content = OutBlock(32, 3)
        
        self.dblock1_attention = DBlock(128, 32, 2, 1)
        self.dblock2_attention = DBlock(64, 16, 2, 1)
        self.outblock_attention = OutBlock(32, 3)

        self.mask_weight = torch.nn.Parameter(torch.zeros((1),dtype=torch.float32,requires_grad=True))
        
        self.input_padding = None
        # if xavier_init_all:
        #     for name, m in self.named_modules():
        #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #             torch.nn.init.xavier_normal_(m.weight)
        #             # torch.nn.init.kaiming_normal_(m.weight)
        #             print(name)
        
        print('Net small is created.')

    def forward_step(self, x, hidden_state):
        
        e16 = self.inblock(x)
        e32 = self.eblock1(e16)
        e64 = self.eblock2(e32)
        h, c = self.convlstm(e64, hidden_state)
        
        d32_content = self.dblock1_content(torch.concat([h, e64], dim=1))
        d16_content = self.dblock2_content(torch.concat([d32_content, e32], dim=1))
        d3_content = self.outblock_content(torch.concat([d16_content, e16], dim=1))
        
        d32_attention = self.dblock1_attention(torch.concat([h, e64], dim=1))
        d16_attention = self.dblock2_attention(torch.concat([d32_attention, e32],dim=1))
        d3_attention = self.outblock_attention(torch.concat([d16_attention, e16],dim=1))
        
        
        # d3_attention = torch.nn.functional.softmax(d3_attention, dim=1)
        d3_content = torch.sigmoid(d3_content)
        d3_attention = torch.tanh(d3_attention)
        
        # d3_content = torch.sigmoid(d3_content)
        # d3_attention = torch.sigmoid(d3_attention)
        
        xs = list(torch.split(x, 3, 1))
        if self.style == 1:
            d3 = d3_content * d3_attention * self.mask_weight+ xs[0] *(1-self.mask_weight)
        elif self.style == 2:
            d3 = d3_content * d3_attention + xs[0] * (1 - d3_attention)
        elif self.style == 3:
            d3 = d3_content * d3_attention + xs[0]
        elif self.style == 4:
            d3 = d3_content

        return d3, d3_attention,h,c

    def forward(self, b1, b2, b3, b4):
        if self.mean_shift:
            b1 = b1 - 0.5
            b2 = b2 - 0.5
            b3 = b3 - 0.5
            b4 = b4 - 0.5
            
        b1 = b1 * self.range_of_image
        b2 = b2 * self.range_of_image
        b3 = b3 * self.range_of_image
        b4 = b4 * self.range_of_image
        
        if self.level == 1:
            h, c = self.convlstm.init_hidden(b1.shape[0], (b1.shape[-2]//4, b1.shape[-1]//4))

            i1, a1,h,c = self.forward_step(
                torch.cat([b1, torch.zeros_like(b1)], 1), (h, c))

            i2 = torch.zeros(b2.shape).cuda()
            a2 = torch.zeros(b2.shape).cuda()
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
        
        if self.level == 2:
            h, c = self.convlstm.init_hidden(b2.shape[0], (b2.shape[-2]//4, b2.shape[-1]//4))

            i2, a2,h,c = self.forward_step(
                torch.cat([b2, torch.zeros_like(b2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
            
        if self.level == 3:
            h, c = self.convlstm.init_hidden(b3.shape[0], (b3.shape[-2]//4, b3.shape[-1]//4))

            i3, a3,h,c = self.forward_step(
                torch.cat([b3, torch.zeros_like(b3)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i2, a2,h,c = self.forward_step(
                torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
            
        if self.level == 4:
            h, c = self.convlstm.init_hidden(b4.shape[0], (b4.shape[-2]//4, b4.shape[-1]//4))
        
            i4, a4,h,c = self.forward_step(
                torch.cat([b4, torch.zeros_like(b4)], 1), (h, c))
            
            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            
            i3, a3,h,c = self.forward_step(
                torch.cat([b3, self.upsample_fn(i4, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i2, a2,h,c = self.forward_step(
                torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
        
        i1 = i1 / self.range_of_image
        i2 = i2 / self.range_of_image
        i3 = i3 / self.range_of_image
        i4 = i4 / self.range_of_image
        
        if self.mean_shift:
            i1 = i1 + 0.5
            i2 = i2 + 0.5
            i3 = i3 + 0.5
            i4 = i4 + 0.5
            
        return i1, i2, i3, i4, a1, a2, a3, a4
    
class Net_concat_wide(nn.Module):
    def __init__(self,level = 3,style = 1, mean_shift = True, range_of_image = 1.0,upsample_fn=partial(torch.nn.functional.interpolate, mode='bilinear'), xavier_init_all=True):
        super(type(self), self).__init__()
        self.level = level
        self.style = style
        self.mean_shift = mean_shift
        self.range_of_image = range_of_image
        self.upsample_fn = upsample_fn
        self.inblock = EBlock(3 + 3, 16, 1)
        self.eblock1 = EBlock(16, 32, 2)
        self.eblock2 = EBlock(32, 128, 2)
        
        self.convlstm = CLSTM_cell(128, 128, 5)
        
        self.dblock1_content = DBlock(128, 32, 2, 1)
        self.dblock2_content = DBlock(64, 16, 2, 1)
        self.outblock_content = OutBlock(32, 3)
        
        self.dblock1_attention = DBlock(128, 32, 2, 1)
        self.dblock2_attention = DBlock(64, 16, 2, 1)
        self.outblock_attention = OutBlock(32, 3)

        self.mask_weight = torch.nn.Parameter(torch.zeros((1),dtype=torch.float32,requires_grad=True))
        
        self.input_padding = None
        # if xavier_init_all:
        #     for name, m in self.named_modules():
        #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #             torch.nn.init.xavier_normal_(m.weight)
        #             # torch.nn.init.kaiming_normal_(m.weight)
        #             print(name)
        
        print('Net small is created.')

    def forward_step(self, x, hidden_state):
        
        e16 = self.inblock(x)
        e32 = self.eblock1(e16)
        e64 = self.eblock2(e32)
        h, c = self.convlstm(e64, hidden_state)
        
        d32_content = self.dblock1_content(h)
        d16_content = self.dblock2_content(torch.concat([d32_content, e32], dim=1))
        d3_content = self.outblock_content(torch.concat([d16_content, e16], dim=1))
        
        d32_attention = self.dblock1_attention(h)
        d16_attention = self.dblock2_attention(torch.concat([d32_attention, e32],dim=1))
        d3_attention = self.outblock_attention(torch.concat([d16_attention, e16],dim=1))
        
        
        # d3_attention = torch.nn.functional.softmax(d3_attention, dim=1)
        d3_content = torch.sigmoid(d3_content)
        d3_attention = torch.tanh(d3_attention)
        
        # d3_content = torch.sigmoid(d3_content)
        # d3_attention = torch.sigmoid(d3_attention)
        
        xs = list(torch.split(x, 3, 1))
        if self.style == 1:
            d3 = d3_content * d3_attention * self.mask_weight+ xs[0] *(1-self.mask_weight)
        elif self.style == 2:
            d3 = d3_content * d3_attention + xs[0] * (1 - d3_attention)
        elif self.style == 3:
            d3 = d3_content * d3_attention + xs[0]
        elif self.style == 4:
            d3 = d3_content

        return d3, d3_attention,h,c

    def forward(self, b1, b2, b3, b4):
        if self.mean_shift:
            b1 = b1 - 0.5
            b2 = b2 - 0.5
            b3 = b3 - 0.5
            b4 = b4 - 0.5
            
        b1 = b1 * self.range_of_image
        b2 = b2 * self.range_of_image
        b3 = b3 * self.range_of_image
        b4 = b4 * self.range_of_image
        
        if self.level == 1:
            h, c = self.convlstm.init_hidden(b1.shape[0], (b1.shape[-2]//4, b1.shape[-1]//4))

            i1, a1,h,c = self.forward_step(
                torch.cat([b1, torch.zeros_like(b1)], 1), (h, c))

            i2 = torch.zeros(b2.shape).cuda()
            a2 = torch.zeros(b2.shape).cuda()
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
        
        if self.level == 2:
            h, c = self.convlstm.init_hidden(b2.shape[0], (b2.shape[-2]//4, b2.shape[-1]//4))

            i2, a2,h,c = self.forward_step(
                torch.cat([b2, torch.zeros_like(b2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
            
            i3 = torch.zeros(b3.shape).cuda()
            a3 = torch.zeros(b3.shape).cuda()
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
            
        if self.level == 3:
            h, c = self.convlstm.init_hidden(b3.shape[0], (b3.shape[-2]//4, b3.shape[-1]//4))

            i3, a3,h,c = self.forward_step(
                torch.cat([b3, torch.zeros_like(b3)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i2, a2,h,c = self.forward_step(
                torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
            
            i4 = torch.zeros(b4.shape).cuda()
            a4 = torch.zeros(b4.shape).cuda()
            
        if self.level == 4:
            h, c = self.convlstm.init_hidden(b4.shape[0], (b4.shape[-2]//4, b4.shape[-1]//4))
        
            i4, a4,h,c = self.forward_step(
                torch.cat([b4, torch.zeros_like(b4)], 1), (h, c))
            
            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            
            i3, a3,h,c = self.forward_step(
                torch.cat([b3, self.upsample_fn(i4, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i2, a2,h,c = self.forward_step(
                torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1), (h, c))

            c = self.upsample_fn(c, scale_factor=2)
            h = self.upsample_fn(h, scale_factor=2)
            i1, a1,h,c = self.forward_step(
                torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))
        
        i1 = i1 / self.range_of_image
        i2 = i2 / self.range_of_image
        i3 = i3 / self.range_of_image
        i4 = i4 / self.range_of_image
        
        if self.mean_shift:
            i1 = i1 + 0.5
            i2 = i2 + 0.5
            i3 = i3 + 0.5
            i4 = i4 + 0.5
            
        return i1, i2, i3, i4, a1, a2, a3, a4