"""Calibration and sensitivity-aware finetuning of the quantized SR network.

Follows Algorithm 1 of "Outlier-Aware Post-Training Quantization for Image
Super-Resolution" (ICCV 2025): a single calibration epoch over the unlabelled
calibration set, followed by a finetuning phase that cycles through the three
groups of quantization parameters (weight bounds, activation bounds,
breakpoints) and is supervised only by the full-precision teacher.
"""

import os
import math
import time
import datetime
import shutil

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs

from decimal import Decimal
from tqdm import tqdm
import cv2

import utility
from model.quantize import QConv2d

class Trainer():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_init = loader.loader_init
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.epoch = 0
        # Layer-wise quantization sensitivities s_k (Eq. 4), filled in during calibration.
        self.layer_sensitivity = []
        shutil.copyfile('./trainer.py', os.path.join(self.ckp.dir, 'trainer.py'))
        shutil.copyfile('./model/quantize.py', os.path.join(self.ckp.dir, 'quantize.py'))

        quant_params_a = [v for k, v in self.model.model.named_parameters() if '_a' in k]
        quant_params_w = [v for k, v in self.model.model.named_parameters() if '_w' in k]
        quant_params_bp = [v for k, v in self.model.model.named_parameters() if '_bp' in k]

        self.optimizer_a = torch.optim.Adam(quant_params_a, lr=args.lr_a, betas=args.betas, eps=args.epsilon)
        self.optimizer_w = torch.optim.Adam(quant_params_w, lr=args.lr_w, betas=args.betas, eps=args.epsilon)
        self.optimizer_bp = torch.optim.Adam(quant_params_bp, lr=args.lr_bp, betas=args.betas, eps=args.epsilon)
        self.scheduler_a = lrs.StepLR(self.optimizer_a, step_size=args.step, gamma=args.gamma)
        self.scheduler_w = lrs.StepLR(self.optimizer_w, step_size=args.step, gamma=args.gamma)
        self.scheduler_bp = lrs.StepLR(self.optimizer_bp, step_size=args.step, gamma=args.gamma)
        
        self.sen_losses = utility.AverageMeter()
        self.pix_losses = utility.AverageMeter()
        self.bit_losses = utility.AverageMeter()
        self.measure_layer_max_list = []
        self.measure_layer_min_list = []


        self.num_quant_modules = 0
        for n, m in self.model.named_modules():
            if isinstance(m, QConv2d):
                if not m.to_8bit: # 8-bit (first or last) modules are excluded for the bit count
                    self.num_quant_modules +=1
        
        # for initialization
        if not args.test_only:
            for n, m in self.model.named_modules():
                if isinstance(m, QConv2d):
                    setattr(m, 'w_bit', 32.0)
                    setattr(m, 'a_bit', 32.0)
                    setattr(m, 'init', True)
    
    def get_stage_optimizer_scheduler(self):
        """Pick the parameter group to update this epoch (Algorithm 1, l.17-24).

        The three groups are refreshed in turn: weight upper bounds u_w,
        activation bounds (l_a, u_a), then the breakpoints bp.
        """
        if (self.epoch - 1) % 3 == 0:
            param_name = '_w'
            optimizer, scheduler = self.optimizer_w, self.scheduler_w
        elif (self.epoch - 1) % 3 == 1:
            param_name = '_a'
            optimizer, scheduler = self.optimizer_a, self.scheduler_a
        else:
            param_name = '_bp'
            optimizer, scheduler = self.optimizer_bp, self.scheduler_bp

        return param_name, optimizer, scheduler
    
    def set_bit(self, teacher=False):
        for n, m in self.model.named_modules():
            if isinstance(m, QConv2d):
                if teacher:
                    setattr(m, 'w_bit', 32.0)
                    setattr(m, 'a_bit', 32.0)
                else:
                    if m.to_8bit:
                        setattr(m, 'w_bit', 8.0)
                        setattr(m, 'a_bit', 8.0)
                    else:
                        setattr(m, 'w_bit', self.args.quantize_w)
                        setattr(m, 'a_bit', self.args.quantize_a)

                setattr(m, 'init', False)

    def train(self):
        if self.epoch > 0:
            param_name, optimizer, scheduler = self.get_stage_optimizer_scheduler()
            epoch_update = 'Update param ' + param_name
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            self.ckp.write_log(
                '\n[Epoch {}]\t {}\t Learning rate for param: {:.2e}'.format(
                    self.epoch,
                    epoch_update,
                    Decimal(lr))
            )
            
        self.model.train()
        
        timer_data, timer_model = utility.timer(), utility.timer()
        start_time = time.time()
        
        if self.epoch == 0:
            # Initialize Q parameters using freezed FP model
            params = self.model.named_parameters()
            for name1, params1 in params:
                params1.requires_grad=False
            
            for batch, (lr, _, idx_scale,) in enumerate(self.loader_init):
                lr, = self.prepare(lr)

                timer_data.hold()
                timer_model.tic()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    sr_temp, feat_temp, bit_temp, err = self.model(lr, idx_scale) #?
                display_bit = bit_temp.mean() / self.num_quant_modules
                
                if self.args.fine_tuning:
                    self.ckp.write_log('[{}/{}] [bit:{:.2f}] \t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.loader_init.batch_size,
                        len(self.loader_init.dataset),
                        display_bit,
                        timer_model.release(),
                        timer_data.release(), 
                    ))

            if self.args.count_std:
                # Quantization sensitivity s_k of every quantized layer (Eq. 4):
                # a softmax over the mean feature standard deviation collected
                # during the calibration pass.
                mean_std = []
                for _, m in self.model.named_modules():
                    if isinstance(m, QConv2d):
                        if hasattr(m, 'std_layer') and len(m.std_layer) > 0:
                            mean_std.append(np.mean(m.std_layer))

                weights = [np.exp(v) for v in mean_std]
                total = sum(weights)
                self.layer_sensitivity = [v / total for v in weights]

            print('Calibration done!')

            if self.args.fine_tuning:
                bit_layer_list = []
                for n, m in self.model.named_modules():
                    if isinstance(m, QConv2d):
                        bit_layer_list.append(int(m.measure_layer.data.item()))
                print(bit_layer_list)
                print(np.mean(bit_layer_list)) 
        
        else: # elif self.epoch >0 and self.epoch < self.args.epochs:
            # Update quantization parameters
            for k, v in self.model.named_parameters():
                if param_name in k:
                    v.requires_grad=True
                else:
                    v.requires_grad=False
            
            self.bit_losses.reset()
            self.pix_losses.reset()
            self.sen_losses.reset()

            for batch, (lr, _, idx_scale,) in enumerate(self.loader_train):
                lr, = self.prepare(lr)
                timer_data.hold()
                timer_model.tic()

                optimizer.zero_grad()

                with torch.no_grad():
                    self.set_bit(teacher=True)
                    sr_t, feat_t, bit_t, err = self.model(lr, idx_scale)
                self.set_bit(teacher=False)
                sr, feat, bit, err = self.model(lr, idx_scale)

                # Reconstruction loss (Eq. 5): the full-precision output is the
                # only supervision, so no ground-truth HR image is needed.
                rec_loss = F.l1_loss(sr, sr_t)

                # Sensitivity-aware loss (Eq. 6): the distance between the
                # L2-normalised teacher / student features, weighted by the
                # quantization sensitivity of the corresponding layer.
                sen_loss = torch.tensor(0.0).cuda()
                for block in range(len(feat)):
                    f_q = feat[block] / torch.norm(feat[block], p=2)
                    f_k = feat_t[block] / torch.norm(feat_t[block], p=2)
                    weight = self.layer_sensitivity[block] if self.args.saft else 1.0
                    sen_loss = sen_loss + weight * self.args.w_sktloss \
                        * torch.norm(f_k - f_q, p=2) / sr.shape[0] / len(feat)

                # Total objective (Eq. 7).
                loss = sen_loss + self.args.lambda_rec * rec_loss
                loss.backward()

                self.pix_losses.update(rec_loss.item(), lr.size(0))
                display_pix_loss = f'L_rec: {self.pix_losses.avg: .3f}'
                self.sen_losses.update(sen_loss.item(), lr.size(0))
                display_skt_loss = f'L_sen: {self.sen_losses.avg: .3f}'

                optimizer.step()
                timer_model.hold()

                if (batch + 1) % self.args.print_every == 0:
                    display_bit = bit.mean() / self.num_quant_modules
                    if self.args.fine_tuning:
                        self.ckp.write_log('[{}/{}]\t{} \t{} [bit:{:.2f}] \t{:.1f}+{:.1f}s'.format(
                            (batch + 1) * self.loader_train.batch_size,
                            len(self.loader_train.dataset),
                            display_pix_loss,
                            display_skt_loss,
                            display_bit,
                            timer_model.release(),
                            timer_data.release(), 
                        ))
                    else:
                        self.ckp.write_log('[{}/{}]\t{} \t{} [bit:{:.2f}] \t{:.1f}+{:.1f}s'.format(
                            (batch + 1) * self.loader_train.batch_size,
                            len(self.loader_train.dataset),
                            display_pix_loss,
                            display_skt_loss,
                            display_bit,
                            timer_model.release(),
                            timer_data.release(), 
                        ))
                timer_data.tic()
          
            scheduler.step()      
        self.epoch += 1

        end_time = time.time()
        time_interval = end_time - start_time
        t_string = "Epoch Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
        self.ckp.write_log('{}'.format(t_string))
    
    def patch_inference(self, model, lr, idx_scale):
        patch_idx = 0
        tot_bit_image = 0
        if self.args.n_parallel!=1: 
            lr_list, num_h, num_w, h, w = utility.crop_parallel(lr, self.args.test_patch_size, self.args.test_step_size)
            sr_list = torch.Tensor().cuda()
            for lr_sub_index in range(len(lr_list)// self.args.n_parallel + 1):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    sr_sub, feat, bit = self.model(lr_list[lr_sub_index* self.args.n_parallel: (lr_sub_index+1)*self.args.n_parallel], idx_scale)
                    sr_sub = utility.quantize(sr_sub, self.args.rgb_range)
                sr_list = torch.cat([sr_list, sr_sub])
                average_bit = bit.mean() / self.num_quant_modules
                tot_bit_image += average_bit
                patch_idx += 1
            sr = utility.combine(sr_list, num_h, num_w, h, w, self.args.test_patch_size, self.args.test_step_size, self.scale[0])
        else:
            lr_list, num_h, num_w, h, w = utility.crop(lr, self.args.test_patch_size, self.args.test_step_size)
            sr_list = []
            for lr_sub_img in lr_list:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    sr_sub, feat, bit, err = self.model(lr_sub_img, idx_scale)
                    sr_sub = utility.quantize(sr_sub, self.args.rgb_range)
                sr_list.append(sr_sub)
                average_bit = bit.mean() / self.num_quant_modules
                tot_bit_image += average_bit
                patch_idx += 1
            sr = utility.combine(sr_list, num_h, num_w, h, w, self.args.test_patch_size, self.args.test_step_size, self.scale[0])

        bit = tot_bit_image / patch_idx

        return sr, feat, bit

    def test(self):
        torch.set_grad_enabled(False)
        # if True:
        if self.epoch > 1 or self.args.test_only:
            self.ckp.write_log('\nEvaluation:')
            self.ckp.add_log(
                torch.zeros(1, len(self.loader_test), len(self.scale))
            )
            self.model.eval()
            timer_test = utility.timer()
        
            if self.epoch == 2 or self.args.test_only:
                ################### Num of Params, Storage Size ####################
                n_params = 0
                n_params_q = 0
                for k, v in self.model.named_parameters():
                    nn = np.prod(v.size())
                    n_params += nn

                    if 'weight' in k:
                        name_split = k.split(".")
                        del name_split[-1]
                        module_temp = self.model
                        for n in name_split:
                            module_temp = getattr(module_temp, n)
                        if isinstance(module_temp, QConv2d):
                            n_params_q += nn * module_temp.w_bit / 32.0
                            # print(k, module_temp.w_bit)
                        else:
                            n_params_q += nn
                    else:
                        n_params_q += nn

                self.ckp.write_log('Parameters: {:.3f}K'.format(n_params/(10**3)))
                self.ckp.write_log('Model Size: {:.3f}K'.format(n_params_q/(10**3)))
        
            if self.args.save_results:
                self.ckp.begin_background()
        
            ############################## TEST FOR OWN #############################
            if self.args.test_own is not None:
                test_img = cv2.imread(self.args.test_own)
                lr = torch.tensor(test_img).permute(2,0,1).float().cuda()
                lr = torch.flip(lr, (0,)) # for color
                lr = lr.unsqueeze(0)

                tot_bit = 0
                for idx_scale, scale in enumerate(self.scale):
                    if self.args.test_patch:
                        sr, feat, bit = self.patch_inference(self.model, lr, idx_scale)
                        img_bit = bit
                    else:
                        with torch.no_grad():
                            sr, feat, bit = self.model(lr, idx_scale)
                        img_bit = bit.mean() / self.num_quant_modules

                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]


                    filename = self.args.test_own.split('/')[-1].split('.')[0]
                    if self.args.save_results:
                        save_name = '{}_x{}_{:.2f}bit'.format(filename, scale, img_bit)
                        self.ckp.save_results('test_own', save_name, save_list)

                    self.ckp.write_log('[{} x{}] Average Bit: {:.2f} '.format(filename, scale, img_bit))

            ############################## TEST FOR TEST SET #############################
            if self.args.test_own is None:
                for idx_data, d in enumerate(self.loader_test):
                    for idx_scale, scale in enumerate(self.scale):
                        d.dataset.set_scale(idx_scale)
                        tot_ssim =0
                        tot_bit =0 
                        i=0
                        bitops =0
                        for lr, hr, filename in tqdm(d, ncols=80):
                            i+=1
                            lr, hr = self.prepare(lr, hr)
                        
                            if self.args.test_patch:
                                sr, feat, bit = self.patch_inference(self.model, lr, idx_scale)
                                if self.args.n_parallel!=1: hr = hr[:, :, :lr.shape[2]*self.scale[0], :lr.shape[2]*self.scale[0]] 
                                img_bit = bit.item()
                            else:
                                with torch.no_grad():
                                    sr, feat, bit, err = self.model(lr, idx_scale)
                                img_bit = bit.mean().item() / self.num_quant_modules


                            sr = utility.quantize(sr, self.args.rgb_range)
                            save_list = [sr]

                            psnr, ssim = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)

                            self.ckp.ssim_log[-1, idx_data, idx_scale] += ssim
                            self.ckp.log[-1, idx_data, idx_scale] += psnr
                            self.ckp.bit_log[-1, idx_data, idx_scale] += img_bit
                            
                            if self.args.save_gt:
                                save_list.extend([lr, hr])

                            if self.args.save_results:
                                save_name = '{}_x{}_{:.2f}dB_{:.2f}bit'.format(filename[0], scale, psnr, img_bit)
                                self.ckp.save_results(d, save_name, save_list)
                            
                        self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                        self.ckp.ssim_log[-1, idx_data, idx_scale] /= len(d)
                        self.ckp.bit_log[-1, idx_data, idx_scale] /= len(d)

                        best = self.ckp.log.max(0)
                        self.ckp.write_log(
                            '[{} x{}]\tPSNR: {:.3f} \t SSIM: {:.4f} \tBit: {:.2f} \t(Best: {:.3f} @epoch {})'.format(
                                d.dataset.name,
                                scale,
                                self.ckp.log[-1, idx_data, idx_scale],
                                self.ckp.ssim_log[-1, idx_data, idx_scale],
                                self.ckp.bit_log[-1, idx_data, idx_scale],
                                best[0][idx_data, idx_scale],
                                best[1][idx_data, idx_scale] + 1
                            )
                        )
                        
            if self.args.save_results:
                self.ckp.end_background()
            
            # save models
            if not self.args.test_only:
                self.ckp.save(self, self.epoch, is_best=(best[1][0, 0] + 1 == self.epoch -1))

        torch.set_grad_enabled(True) 

    def test_teacher(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        self.ckp.write_log('Teacher Evaluation')

        ############################## Num of Params ####################
        n_params = 0
        for k, v in self.model.named_parameters():
            if '_a' not in k and '_w' not in k and '_cp' not in k: # for teacher model
                n_params += np.prod(v.size())
        self.ckp.write_log('Parameters: {:.3f}K'.format(n_params/(10**3)))

        if self.args.save_results:
            self.ckp.begin_background()
        
        ############################## TEST FOR OWN #############################
        if self.args.test_own is not None:
            test_img = cv2.imread(self.args.test_own)
            lr = torch.tensor(test_img).permute(2,0,1).float().cuda()
            lr = torch.flip(lr, (0,)) # for color
            lr = lr.unsqueeze(0)

            tot_bit = 0
            for idx_scale, scale in enumerate(self.scale):
                self.set_bit(teacher=True)
                if self.args.test_patch:
                    sr, feat, bit = self.patch_inference(self.model, lr, idx_scale)
                    img_bit = bit
                else:
                    with torch.no_grad():
                        sr, feat, bit = self.model(lr, idx_scale)
                    img_bit = bit.mean() / self.num_quant_modules                    
                self.set_bit(teacher=False)

                sr = utility.quantize(sr, self.args.rgb_range)
                save_list = [sr]
                if self.args.save_results:
                    filename = self.args.test_own.split('/')[-1].split('.')[0]
                    save_name = '{}_x{}_{:.2f}bit'.format(filename, scale, img_bit)
                    self.ckp.save_results('test_own', save_name, save_list)

        ############################## TEST FOR TEST SET #############################
        if self.args.test_own is None:
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    tot_ssim =0
                    tot_bit =0 
                    tot_psnr =0.0
                    i=0
                    for lr, hr, filename in tqdm(d, ncols=80):
                        i+=1
                        lr, hr = self.prepare(lr, hr)
                        self.set_bit(teacher=True)
                        if self.args.test_patch:
                            sr, feat, bit = self.patch_inference(self.model, lr, idx_scale)
                            img_bit = bit
                        else:
                            with torch.no_grad():
                                sr, feat, bit, err = self.model(lr, idx_scale)
                            img_bit = bit.mean() / self.num_quant_modules
                        self.set_bit(teacher=False)

                        sr = utility.quantize(sr, self.args.rgb_range)
                        save_list = [sr]
                        psnr, ssim = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)

                        tot_bit += img_bit
                        tot_psnr += psnr
                        tot_ssim += ssim

                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            save_name = '{}_x{}_{:.2f}dB'.format(filename[0], scale, cur_psnr)
                            self.ckp.save_results(d, save_name, save_list)

                    tot_psnr /= len(d)
                    tot_ssim /= len(d)
                    tot_bit /= len(d)

                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} \t SSIM: {:.4f} \tBit: {:.2f}'.format(
                            d.dataset.name,
                            scale,
                            tot_psnr,
                            tot_ssim,
                            tot_bit.item(),
                        )
                    )

        if self.args.save_results:
            self.ckp.end_background()

        torch.set_grad_enabled(True)



    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            # return self.epoch >= self.args.epochs
            return self.epoch > self.args.epochs
