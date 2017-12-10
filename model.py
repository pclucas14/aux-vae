import pdb
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import * 
from utils.layers import * 
import numpy as np

class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=0) 
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=2) 
                                            for _ in range(nr_resnet)])
        
        # stream from vae output
        self.vae_stream = nn.ModuleList([gated_resnet(nr_filters, conv2d_norm, 
                                        resnet_nonlinearity, skip_connection=0) 
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, vae):
        u_list, ul_list, vae_list = [], [], []
        
        for i in range(self.nr_resnet):
            u   = self.u_stream[i](u)
            vae = self.vae_stream[i](vae)
            ul  = self.ul_stream[i](ul, a=torch.cat((vae, u), 1))
            u_list   += [u]
            vae_list += [vae]
            ul_list  += [ul]

        return u_list, ul_list, vae_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=3) 
                                            for _ in range(nr_resnet)])
        
        # stream from vae output
        self.vae_stream = nn.ModuleList([gated_resnet(nr_filters, conv2d_norm, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, vae, u_list, ul_list, vae_list):
        for i in range(self.nr_resnet):
            u   = self.u_stream[i](u, a=u_list.pop())
            vae = self.vae_stream[i](vae, vae_list.pop())
            ul  = self.ul_stream[i](ul, a=torch.cat((u, vae, ul_list.pop()), 1))
        
        return u, ul, vae
         

class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, 
                    resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' : 
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else : 
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters, 
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters, 
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream   = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters, 
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream  = nn.ModuleList([down_right_shifted_conv2d(nr_filters, 
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.downsize_vae_stream = nn.ModuleList([conv2d_norm(nr_filters, nr_filters, stride=2)
                                                    for _ in range(2)])
        
        self.upsize_u_stream   = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters, 
                                                    stride=(2,2)) for _ in range(2)])
        
        self.upsize_ul_stream  = nn.ModuleList([down_right_shifted_deconv2d(nr_filters, 
                                                    nr_filters, stride=(2,2)) for _ in range(2)])
        
        self.upsize_vae_stream  = nn.ModuleList([deconv2d_norm(nr_filters, nr_filters, stride=2)                                                    
                                                    for _ in range(2)])
        
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3), 
                        shift_output_down=True, norm='weight_norm')

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters, 
                                            filter_size=(1,3), shift_output_down=True, 
                                                norm='weight_norm'), 
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters, 
                                            filter_size=(2,1), shift_output_right=True, 
                                                norm='weight_norm')])
        self.vae_init = conv2d_norm(3, nr_filters, 5)

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None


    def forward(self, x, vae_out=None, sample=False):
        # similar as done in the tf repo :  
        if self.init_padding is None and not sample: 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding
        
        if sample : 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list   = [self.u_init(x)]
        ul_list  = [self.ul_init[0](x) + self.ul_init[1](x)]
        vae_list = [self.vae_init(vae_out)]
        for i in range(3):
            # resnet block
            u_out, ul_out, vae_out = self.up_layers[i](u_list[-1], ul_list[-1], vae_list[-1])
            u_list   += u_out
            ul_list  += ul_out
            vae_list += vae_out

            if i != 2: 
                # downscale (only twice)
                u_list   += [self.downsize_u_stream[i](u_list[-1])]
                ul_list  += [self.downsize_ul_stream[i](ul_list[-1])]
                vae_list += [self.downsize_vae_stream[i](vae_list[-1])]

        ###    DOWN PASS    ###
        u   = u_list.pop()
        ul  = ul_list.pop()
        vae = vae_list.pop()
        
        for i in range(3):
            # resnet block
            u, ul, vae = self.down_layers[i](u, ul, vae, u_list, ul_list, vae_list)

            # upscale (only twice)
            if i != 2 :
                u   = self.upsize_u_stream[i](u)
                ul  = self.upsize_ul_stream[i](ul)
                vae = self.upsize_vae_stream[i](vae)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
        
