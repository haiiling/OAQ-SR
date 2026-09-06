"""Entry point: calibrate and finetune a quantized SR network.

    Outlier-Aware Post-Training Quantization for Image Super-Resolution
    Hailing Wang, Jianglin Lu, Yitian Zhang, Yun Fu (ICCV 2025)
"""

import datetime
import random
import time

import numpy
import torch

import data
import model
import utility
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    exp_start_time = time.time()

    _loader = data.Data(args)
    _model = model.Model(args, checkpoint)
    t = Trainer(args, _loader, _model, checkpoint)

    while not t.terminate():
        torch.manual_seed(args.seed)
        t.train()
        t.test()

    elapsed = time.time() - exp_start_time
    checkpoint.write_log('Total Running Time is: {}\n'.format(
        datetime.timedelta(seconds=elapsed)))
    checkpoint.done()
