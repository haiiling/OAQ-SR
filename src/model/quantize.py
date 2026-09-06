"""Quantized convolution with a piecewise linear activation quantizer.

Implements the outlier-aware quantizer of

    Outlier-Aware Post-Training Quantization for Image Super-Resolution
    Hailing Wang, Jianglin Lu, Yitian Zhang, Yun Fu (ICCV 2025)

The activation range [l_a, u_a] is split at a breakpoint bp into a symmetric
dense region [-bp, bp] and an outlier region [l_a, -bp) U (bp, u_a].  Each
region is quantized uniformly and independently (Eq. 3 in the paper), so that
outliers are preserved without consuming the bit-width reserved for the bulk
of the distribution.
"""

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm, laplace

from .pwlq import *


class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x_out = torch.round(x_in)
        return x_out
    @staticmethod
    def backward(ctx, g):
        return g, None



class QConv2d(nn.Conv2d):
    def __init__(self, args, in_channels, out_channels, kernel_size, stride=1, 
                padding=1, bias=False, dilation=1, groups=1, to_8bit=False):
        super(QConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation, groups=groups)
        self.args = args
        #if not bias: self.bias = None
        self.dilation = (dilation, dilation)

        # For quantizing activations
        self.lower_a = nn.Parameter(torch.FloatTensor([-256]).cuda())
        self.upper_a = nn.Parameter(torch.FloatTensor([256]).cuda())
        self.break_point_bp = nn.Parameter(torch.FloatTensor([50]).cuda())#, requires_grad=False
        self.round_a = Round.apply
        self.a_bit = self.args.quantize_a
        
        # For quantizing weights
        self.upper_w = nn.Parameter(torch.FloatTensor([128]).cuda())
        self.round_w = Round.apply
        self.w_bit = self.args.quantize_w

        self.to_8bit = to_8bit

        if self.to_8bit:
            self.w_bit = 8.0
            self.a_bit = 8.0
        
        if self.args.count_std:
            self.std_layer = []

        self.ema_epoch = 1
        self.bac_epoch = 1
        self.init = False
    def compute_entropy(self, hist):
        """Shannon entropy of a histogram."""
        prob = hist / hist.sum()
        prob = prob[prob > 0]                        # guard against log(0)
        entropy = -torch.sum(prob * torch.log2(prob))
        return entropy

    def derivative_quant_err(self, m, p, dist='norm', pw_opt=2):  
        '''
        Compute the derivative of expected variance of quantization error
        '''
        if dist == 'norm':
            cdf_func = norm.cdf(p)
            pdf_func = norm.pdf(p)
        elif dist == 'laplace':  
            # https://en.wikipedia.org/wiki/Laplace_distribution
            cdf_func = laplace.cdf(p, 0, np.sqrt(0.5))   
            pdf_func = laplace.pdf(p, 0, np.sqrt(0.5)) # pdf(p, a, b) has variance 2*b^2
        else:
            raise RuntimeError("Not implemented for distribution: %s !!!" % dist) 
        
        ## option 1: overlapping
        if pw_opt == 1: 
            # quant_err = [F(p) - F(-p)] * p^2 + 2*[F(m) - F(p)] * m^2
            df_dp = 2 * pdf_func * (p * p - m * m) + 2 * p * (2 * cdf_func - 1.0)
        ## option 2: non-overlapping
        else:  
            # quant_err = [F(p) - F(-p)] * p^2 + 2*[F(m) - F(p)] * (m - p)^2
            df_dp = p - 2 * m + 2 * m * cdf_func + m * pdf_func * (2 * p - m) 

        return df_dp
    
    def binary_search(self, m, pw_opt, dist, max_iter=100, tol=1e-3):
        '''
        Binary search method to find the optimal breakpoint
        '''
        left, right = 0, m
        fl = self.derivative_quant_err(m, left, pw_opt=pw_opt, dist=dist)
        fr = self.derivative_quant_err(m, right, pw_opt=pw_opt, dist=dist)
        err, iter_num = 1, 0
        while err > tol and iter_num < max_iter:
            mid = (right - left) / 2.0 + left
            fm = self.derivative_quant_err(m, mid, pw_opt=pw_opt, dist=dist)
            if fm * fl > 0:
                left = mid
                fl = fm
            else:
                right = mid
                fr = fm
            err = np.abs(fm)
            iter_num += 1
        return mid
    def pwlq_quant_error(self, w, bits, scale_bits, abs_max, break_point):
        '''
        Piecewise linear quantization (PWLQ) with two options: overlapping or non-overlapping
        '''
        qw_tail_neg = self.uniform_affine_quantizer(w, 
            bits=bits-1, scale_bits=scale_bits, minv=-abs_max, maxv=-break_point)
        qw_tail_pos = self.uniform_affine_quantizer(w, 
            bits=bits-1, scale_bits=scale_bits, minv=break_point, maxv=abs_max)
        qw_middle = self.uniform_symmetric_quantizer(w, 
            bits=bits, scale_bits=scale_bits, minv=-break_point, maxv=break_point)
    
        qw = torch.where(-break_point < w, qw_middle, qw_tail_neg)
        qw = torch.where(break_point > w, qw, qw_tail_pos)

        err = torch.sqrt(torch.sum(torch.mul(qw - w, qw - w)))
        return err, qw
    def init_qparams_a(self, x, quantizer=None, bp_init=None):
        # Obtain statistics
        if quantizer == 'minmax':
            lower_a = torch.min(x).detach().cpu() #x.amax(dim=(0,2,3)).detach().cpu() 
            upper_a = torch.max(x).detach().cpu() #x.amin(dim=(0,2,3)).detach().cpu() 

        elif quantizer == 'percentile':
            try:
                lower_a = torch.quantile(x.reshape(-1), 1.0-self.args.percentile_alpha).detach().cpu()
                upper_a = torch.quantile(x.reshape(-1), self.args.percentile_alpha).detach().cpu()
            except:
                lower_a = np.percentile(x.reshape(-1).detach().cpu(), (1.0-self.args.percentile_alpha)*100.0)
                upper_a = np.percentile(x.reshape(-1).detach().cpu(), self.args.percentile_alpha*100.0)

        elif quantizer == 'omse':
            lower_a = torch.min(x).detach()
            upper_a = torch.max(x).detach()
            best_score = 1e+10
            for i in range(90):
                new_lower = lower_a * (1.0 - i*0.01)
                new_upper = upper_a * (1.0 - i*0.01)
                x_q = torch.clamp(x, min= new_lower, max=new_upper)
                x_q = (x_q - new_lower) / (new_upper - new_lower)
                x_q = torch.round(x_q * (2**self.args.quantize_a -1)) / (2**self.args.quantize_a -1)
                x_q = x_q * (new_upper - new_lower) + new_lower
                score = (x - x_q).abs().pow(2.0).mean()
                if score < best_score:
                    best_score = score
                    best_lower = new_lower
                    best_upper = new_upper
            lower_a = best_lower.cpu()
            upper_a = best_upper.cpu()
        elif quantizer == "entropy":
            print("entropy")
            lower_a = torch.min(x).detach().cpu()
            upper_a = torch.max(x).detach().cpu()

            # Histogram of the incoming activations.
            num_bins = 256
            hist = torch.histc(x, bins=num_bins, min=lower_a.item(), max=upper_a.item())
            bin_edges = torch.linspace(lower_a, upper_a, steps=num_bins + 1)
            best_entropy = float('-inf')
            best_lower = lower_a
            best_upper = upper_a

            # Sweep candidate bounds and keep the range with maximum entropy.
            for i in range(90):
                new_lower = lower_a * (1.0 - i * 0.01)
                new_upper = upper_a * (1.0 - i * 0.01)

                # Clip the histogram to the candidate bounds.
                clipped_hist = hist[(bin_edges[:-1] >= new_lower) & (bin_edges[1:] <= new_upper)]
                if clipped_hist.sum() == 0:
                    continue                          # no activations left after clipping

                entropy = self.compute_entropy(clipped_hist)
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_lower = new_lower
                    best_upper = new_upper
            lower_a = best_lower.cpu()
            upper_a = best_upper.cpu()
        std_x = torch.std(x) + 1e-12
        abs_max = torch.max(torch.abs(x)) 
        abs_max_normalized = (abs_max / std_x).cpu().numpy()
        if bp_init == "norm":
            # Approximated version for Gaussian
            coef = 0.86143114  
            inte = 0.607901097496529 
            break_point = np.log(coef * abs_max_normalized + inte)
            bkp_ratio = break_point / abs_max_normalized
            break_point = bkp_ratio * abs_max
        elif bp_init == "laplace":
            # Approximated version for Laplacian
            coef = 0.80304483
            inte = -0.3166785508381478
            break_point = coef * np.sqrt(abs_max_normalized) + inte
            bkp_ratio = break_point / abs_max_normalized
            break_point = bkp_ratio * abs_max
        elif bp_init == "asym":
            # Paper default: the breakpoint is the 99th percentile of the
            # activations, so that the dense region covers the bulk of the
            # distribution and the tails are treated as outliers.
            data = x.detach().cpu().numpy()
            break_point = np.quantile(data, self.args.percentile_alpha)

        elif bp_init == "search":
            break_point = self.binary_search(abs_max_normalized, pw_opt=2, dist='laplace')
            bkp_ratio = break_point / abs_max_normalized
            break_point = bkp_ratio * abs_max
        elif bp_init == "coarse2fine":
            min_err = 10000
            search_range = 10
            best_ratio = 0
            ## first stage
            for bkp_ratio in np.arange(0.1, 1.0, 0.1):
                break_point = bkp_ratio * abs_max
                err, qw = self.pwlq_quant_error(x, self.args.quantize_a, 0, abs_max, break_point)
                if err < min_err:
                    min_err = err
                    best_ratio = bkp_ratio

            ## second stage
            ratio_start, ratio_end = best_ratio - 0.01 * search_range, best_ratio + 0.01 * search_range
            for bkp_ratio in np.arange(ratio_start, ratio_end, 0.01):
                break_point = bkp_ratio * abs_max
                err, qw = self.pwlq_quant_error(x, self.args.quantize_a, 0, abs_max, break_point)
                if err < min_err:
                    min_err = err
                    best_ratio = bkp_ratio

            ## third stage
            ratio_start, ratio_end = best_ratio - 0.001 * search_range, best_ratio + 0.001 * search_range
            for bkp_ratio in np.arange(ratio_start, ratio_end, 0.001):
                break_point = bkp_ratio * abs_max
                err, qw = self.pwlq_quant_error(x, self.args.quantize_a, 0, abs_max, break_point)
                if err < min_err:
                    min_err = err
                    best_ratio = bkp_ratio
            break_point = best_ratio * abs_max

        else:
            break_point = 50

        # Update q params
        if self.ema_epoch == 1:
            nn.init.constant_(self.lower_a, lower_a)
            nn.init.constant_(self.upper_a, upper_a)
            nn.init.constant_(self.break_point_bp, break_point)
        else:
            beta = self.args.ema_beta
            lower_a = lower_a * (1-beta) + self.lower_a  * beta
            upper_a = upper_a * (1-beta) + self.upper_a  * beta
            break_point = break_point * (1-beta) + break_point * beta
            nn.init.constant_(self.lower_a, lower_a.item())
            nn.init.constant_(self.upper_a, upper_a.item())
            nn.init.constant_(self.break_point_bp, break_point)
    

    def init_qparams_w(self, w, quantizer=None):
        if quantizer == 'minmax':
            upper_w = torch.max(torch.abs(self.weight)).detach()
        elif quantizer == 'percentile':
            try:
                upper_w = torch.quantile(self.weight.reshape(-1), self.args.percentile_alpha).detach().cpu()
            except:
                upper_w = np.percentile(self.weight.reshape(-1).detach().cpu(), self.args.percentile_alpha*100.0)
        elif quantizer == 'omse':
            upper_w = torch.max(self.weight).detach()
            lower_w = torch.min(self.weight).detach()
            best_score_w = 1e+10
            for i in range(90):
                new_upper_w = upper_w * (1.0 - i*0.01)
                new_lower_w = lower_w * (1.0 - i*0.01)
                w_q = torch.clamp(self.weight, min=new_lower_w, max=new_upper_w).detach()
                w_q = (w_q - new_upper_w) / (new_upper_w - new_lower_w)
                w_q = torch.round(w_q * (2**self.args.quantize_w -1)) / (2**self.args.quantize_w -1)
                w_q = w_q * (new_upper_w - new_lower_w) + new_upper_w
                score = (self.weight - w_q).abs().pow(2.0).mean().detach().cpu()
                if score < best_score_w:
                    best_score_w = score
                    best_i = i
                    upper = new_upper_w
                    lower = new_lower_w
            upper_w = upper.cpu()
            lower_w = lower.cpu()
        nn.init.constant_(self.upper_w, upper_w)
    def uniform_symmetric_quantizer(self, x, bits=8.0, minv=None, maxv=None, signed=True, 
                                    scale_bits=0.0, num_levels=None, scale=None, simulated=True):
        if minv is None:
            maxv = torch.max(torch.abs(x))
            minv = - maxv if signed else 0

        if signed:
            maxv = torch.max(minv, maxv)
            minv = -maxv
        else:
            minv = 0
        
        if num_levels is None:
            num_levels = 2 ** bits

        if scale is None:
            scale = (maxv - minv) / (num_levels - 1)

        if scale_bits > 0:
            scale_levels = 2 ** scale_bits
            scale = self.round_a(torch.mul(scale, scale_levels)) / scale_levels
                
        ## clamp
        x = torch.clamp(x, min=float(minv), max=float(maxv))
            
        x_int = self.round_a(x / scale)
        
        if signed:
            x_quant = torch.clamp(x_int, min=-num_levels/2, max=num_levels/2 - 1)
            assert(minv == - maxv)
        else:
            x_quant = torch.clamp(x_int, min=0, max=num_levels - 1)
            assert(minv == 0 and maxv > 0)
            
        x_dequant = x_quant * scale
        
        return x_dequant if simulated else x_quant


    def uniform_affine_quantizer(self, x, bits=8.0, minv=None, maxv=None, offset=None, include_zero=False,
                                scale_bits=0.0, num_levels=None, scale=None, simulated=True):
        if minv is None:
            maxv = torch.max(x)
            minv = torch.min(x)
            if include_zero:
                if minv > 0:
                    minv = 0
                elif maxv < 0:
                    maxv = 0
        
        if num_levels is None:
            num_levels = 2 ** bits
        
        if not scale:
            scale = (maxv - minv) / (num_levels - 1)

        if not offset:
            offset =  minv

        if scale_bits > 0:
            scale_levels = 2 ** scale_bits
            scale = self.round_a(torch.mul(scale, scale_levels)) / scale_levels
            offset = self.round_a(torch.mul(offset, scale_levels)) / scale_levels
            
        ## clamp
        x = torch.clamp(x, min=float(minv), max=float(maxv))
            
        x_int = self.round_a((x - offset) / scale)
        x_quant = torch.clamp(x_int, min=0, max=int(num_levels - 1))
            
        x_dequant = x_quant * scale + offset
        
        return x_dequant if simulated else x_quant
       
    def forward(self, x):
        bit, x = x[1], x[0]
        err = 0

        if self.w_bit == 32:
            # ---- Calibration phase (full-precision pass) ----
            if self.init:
                self.init_qparams_a(x, quantizer=self.args.quantizer,
                                    bp_init=self.args.bp_init)
                if self.ema_epoch == 1:
                    self.init_qparams_w(self.weight, quantizer=self.args.quantizer_w)

                if self.args.count_std:
                    # Layer-wise quantization sensitivity (Eq. 4): the mean
                    # standard deviation of this layer's feature map.
                    measure = lambda t: torch.mean(torch.std(t.detach(), dim=(1, 2, 3)))
                    self.std_layer.append(measure(x).detach().cpu().numpy())

                self.ema_epoch += 1

            a_bit = torch.Tensor([32.0]).cuda()
            w = self.weight

        else:
            a_bit = torch.tensor(self.a_bit).repeat(x.shape[0], 1, 1, 1).cuda()

            # ---- Piecewise linear quantization of activations (Eq. 3) ----
            if self.lower_a < -self.break_point_bp:
                # Two-sided outlier region: the dense region keeps the full
                # bit-width, each tail gets one bit less.
                q_dense = self.uniform_symmetric_quantizer(
                    x, bits=self.a_bit, scale_bits=0,
                    minv=-self.break_point_bp, maxv=self.break_point_bp)
                q_tail_neg = self.uniform_affine_quantizer(
                    x, bits=self.a_bit - 1, scale_bits=0,
                    minv=self.lower_a, maxv=-self.break_point_bp)
                q_tail_pos = self.uniform_affine_quantizer(
                    x, bits=self.a_bit - 1, scale_bits=0,
                    minv=self.break_point_bp, maxv=self.upper_a)

                x_q = torch.where(-self.break_point_bp < x, q_dense, q_tail_neg)
                x_q = torch.where(self.break_point_bp > x, x_q, q_tail_pos)
            else:
                # The lower bound already lies inside the dense region, so a
                # single breakpoint suffices and both sides keep a_bit levels.
                q_low = self.uniform_affine_quantizer(
                    x, bits=self.a_bit, scale_bits=0,
                    minv=self.lower_a, maxv=self.break_point_bp)
                q_high = self.uniform_affine_quantizer(
                    x, bits=self.a_bit, scale_bits=0,
                    minv=self.break_point_bp, maxv=self.upper_a)

                x_q = torch.where(self.break_point_bp < x, q_high, q_low)

            err = torch.sqrt(torch.sum(torch.mul(x_q - x, x_q - x)))
            x = x_q

            # ---- Symmetric uniform quantization of weights ----
            w_c = torch.where(self.weight < -self.upper_w, -self.upper_w, self.weight)
            w_c = torch.where(w_c > self.upper_w, self.upper_w, w_c)
            w_c = (w_c + self.upper_w) / (2 * self.upper_w)
            w_int = self.round_w(w_c * (2 ** self.w_bit - 1)) / (2 ** self.w_bit - 1)
            w = w_int * (2 * self.upper_w) - self.upper_w

        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        out = F.conv2d(x, w, bias=self.bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation, groups=self.groups)
        bit += a_bit.view(-1)
        return out, bit, err
