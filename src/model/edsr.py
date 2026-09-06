from model import common
from model import quantize

import torch.nn as nn
import torch



def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = 'relu'

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.fq = args.fq

        # Head module
        if args.fq:
            m_head = [quantize.QConv2d(args, args.n_colors, n_feats, kernel_size, bias=True, to_8bit=True)]
            
        else:
            m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # Body module
        m_body = [
            common.ResBlock(
                args, conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(args.n_resblocks)
        ]

        if args.fq:
            m_body.append(quantize.QConv2d(args, n_feats, n_feats, kernel_size, bias=True))
        else:
            m_body.append(conv(n_feats, n_feats, kernel_size))

        # Tail module
        if args.fq:
            
            m_tail = [
                common.Upsampler(args, quantize.QConv2d, scale, n_feats, act=False, fq=args.fq), 
                quantize.QConv2d(args, n_feats, args.n_colors, kernel_size, bias=True, to_8bit=True)
            ]
        else:
            m_tail = [
                common.Upsampler(args, conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.args = args


    def forward(self, x):
        err_sum = 0
        x = self.sub_mean(x)
        
        if self.args.fq:
            bit_fq = torch.zeros(x.shape[0]).cuda()
            x, bit_fq, err= self.head([x, bit_fq])

        else:
            x = self.head(x)
 
        feat = None
        bit = torch.zeros(x.shape[0]).cuda()

        res = x

        res, feat, bit, err = self.body[:-1]([res, feat, bit])
        err_sum += err 

        
        if self.args.fq:
            res, bit, err = self.body[-1:]([res, bit])
        else:
            res = self.body[-1:](res)

        res += x

        if self.args.fq:
            
            res, bit_fq, err = self.tail[0][0]([res, bit_fq]) # conv
            res = self.tail[0][1](res) # ps
            if len(self.tail[0]) == 4:
                res, bit_fq, err = self.tail[0][2]([res, bit_fq]) # conv 
                res = self.tail[0][3](res) # ps
            x, bit_fq, err = self.tail[-1]([res, bit_fq]) # conv
        else:
            x = self.tail(res)
            
        x = self.add_mean(x)


        return x, feat, bit, err_sum
    

