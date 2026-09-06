from model import common
import torch
import torch.nn as nn
import math
from model import quantize

def make_model(args, parent=False):
    return SRResNet(args)

class SRResNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRResNet, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]

        # Head module
        self.fq = args.fq
        if args.fq:
            m_head = [quantize.QConv2d(args, args.n_colors, n_feats, kernel_size=9, bias=False, to_8bit=True)]
        else:
            m_head = [conv(args.n_colors, n_feats, kernel_size=9, bias=False)]
        m_head.append(nn.PReLU())

        # Body module
        act = 'prelu'
        m_body = [
            common.ResBlock(
                args, conv, n_feats, kernel_size, bias=True, bn=True, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]

        if args.fq:
            m_body.append(quantize.QConv2d(args, n_feats, n_feats, kernel_size, bias=False))
        else:
            m_body.append(conv(n_feats, n_feats, kernel_size, bias=False))
        m_body.append( nn.BatchNorm2d(n_feats))

        # Tail module
        if args.fq:
            m_tail = [
                common.Upsampler(args, quantize.QConv2d, scale, n_feats, act=act, fq=args.fq, bias=False), 
                quantize.QConv2d(args, n_feats, args.n_colors, kernel_size=9, bias=False, to_8bit=True)
            ]
        else:
            m_tail = [
                common.Upsampler(args, conv, scale, n_feats, act=act, bias=False),
                conv(n_feats, args.n_colors, kernel_size=9, bias=False)
            ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.args = args


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        bit_fq = torch.zeros(x.shape[0]).cuda()

        if self.fq:
            x, bit_fq, err= self.head[0]([x, bit_fq])
            x = self.head[1:](x)
        else:
            x = self.head(x)

        feat = None
        bit = torch.zeros(x.shape[0]).cuda()

        res = x
        

        res, feat, bit, err = self.body[:-2]([res, feat, bit])
        

        if self.fq:
            res, bit, err =self.body[-2]([res, bit])
            res = self.body[-1](res)
        else:
            res = self.body[-2:](res)
        
        
        res+= x

        if self.fq:
            res, bit_fq, err = self.tail[0][0]([res, bit_fq]) # conv
            res = self.tail[0][1](res) # PS
            res = self.tail[0][2](res) # prelu
            res1 =res

            if len(self.tail[0]) > 2:
                res, bit_fq, err = self.tail[0][3]([res, bit_fq]) # conv
                res = self.tail[0][4](res) # PS
                res = self.tail[0][5](res) # prelu
                res2 = res
            x, bit_fq, err = self.tail[-1]([res, bit_fq]) # conv
        else:
            x = self.tail(res)
        
        return x, feat, bit, err
     
