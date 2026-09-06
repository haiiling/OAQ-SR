#!/usr/bin/env bash
#
# Outlier-Aware Post-Training Quantization for Image Super-Resolution (ICCV 2025)
#
# Usage:  sh scripts/run.sh <function> <gpu_id> <a_bit> <w_bit>
#   e.g.  sh scripts/run.sh edsr      0 4 4     # quantize EDSR to W4A4
#         sh scripts/run.sh edsr_eval 0 4 4     # evaluate the quantized model
#
# Run it from the repository root; the commands change into src/ themselves.

set -e
cd "$(dirname "$0")/../src"

DATA_DIR=../datasets/
SCALE=4

# -------------------------------------------------------------------- EDSR ---
edsr() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --model EDSR --scale $SCALE \
        --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
        --pre_train ../pretrained_model/edsr_baseline_x$SCALE.pt \
        --dir_data $DATA_DIR --data_test Set5 \
        --quantize_a $2 --quantize_w $3 --fq \
        --epochs 10 --test_every 50 --print_every 10 \
        --batch_size_calib 16 --batch_size_update 2 --num_data 100 --patch_size 384 \
        --quantizer 'minmax' --quantizer_w 'omse' --bp_init 'asym' \
        --percentile_alpha 0.99 --ema_beta 0.9 \
        --lr_w 0.001 --lr_a 0.001 --lr_bp 0.001 \
        --w_sktloss 10.0 --lambda_rec 5.0 \
        --count_std --saft --bac --layer_percentile 30.0 \
        --save edsr_x$SCALE/w$3a$2 --seed 100
}

edsr_eval() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --model EDSR --scale $SCALE \
        --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
        --dir_data $DATA_DIR --data_test Set5+Set14+B100+Urban100 \
        --quantize_a $2 --quantize_w $3 --fq --test_only \
        --pre_train ../experiment/edsr_x$SCALE/w$3a$2/model/checkpoint_bestpsnr.pt \
        --save edsr_x$SCALE/w$3a$2-eval --save_results
}

# --------------------------------------------------------------------- RDN ---
rdn() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --model RDN --scale $SCALE \
        --pre_train ../pretrained_model/rdn_baseline_x$SCALE.pt \
        --dir_data $DATA_DIR --data_test Set5 \
        --quantize_a $2 --quantize_w $3 --fq \
        --epochs 10 --test_every 50 --print_every 10 \
        --batch_size_calib 16 --batch_size_update 1 --num_data 100 --patch_size 128 \
        --quantizer 'minmax' --quantizer_w 'omse' --bp_init 'asym' \
        --percentile_alpha 0.99 --ema_beta 0.9 \
        --lr_w 0.001 --lr_a 0.001 --lr_bp 0.001 \
        --w_sktloss 10.0 --lambda_rec 5.0 \
        --count_std --saft --bac --layer_percentile 30.0 \
        --save rdn_x$SCALE/w$3a$2 --seed 1
}

rdn_eval() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --model RDN --scale $SCALE \
        --dir_data $DATA_DIR --data_test Set5+Set14+B100+Urban100 \
        --quantize_a $2 --quantize_w $3 --fq --test_only \
        --pre_train ../experiment/rdn_x$SCALE/w$3a$2/model/checkpoint_bestpsnr.pt \
        --save rdn_x$SCALE/w$3a$2-eval --save_results
}

# ---------------------------------------------------------------- SRResNet ---
srresnet() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --model SRResNet --scale $SCALE \
        --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
        --pre_train ../pretrained_model/bnsrresnet_x$SCALE.pt \
        --dir_data $DATA_DIR --data_test Set5 \
        --quantize_a $2 --quantize_w $3 --fq \
        --epochs 10 --test_every 50 --print_every 10 \
        --batch_size_calib 16 --batch_size_update 2 --num_data 100 --patch_size 384 \
        --quantizer 'minmax' --quantizer_w 'omse' --bp_init 'asym' \
        --percentile_alpha 0.99 --ema_beta 0.9 \
        --lr_w 0.001 --lr_a 0.001 --lr_bp 0.001 \
        --w_sktloss 10.0 --lambda_rec 5.0 \
        --count_std --saft --layer_percentile 30.0 \
        --save srresnet_x$SCALE/w$3a$2 --seed 1
}

srresnet_eval() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --model SRResNet --scale $SCALE \
        --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
        --dir_data $DATA_DIR --data_test Set5+Set14+B100+Urban100 \
        --quantize_a $2 --quantize_w $3 --fq --test_only \
        --pre_train ../experiment/srresnet_x$SCALE/w$3a$2/model/checkpoint_bestpsnr.pt \
        --save srresnet_x$SCALE/w$3a$2-eval --save_results
}

# ------------------------------------------------- large inputs (Test2K/4K) ---
edsr_eval_large() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --model EDSR --scale $SCALE \
        --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
        --dir_data $DATA_DIR --data_test test2k+test4k \
        --quantize_a $2 --quantize_w $3 --test_only \
        --test_patch --test_patch_size 96 --test_step_size 96 \
        --pre_train ../experiment/edsr_x$SCALE/w$3a$2/model/checkpoint_bestpsnr.pt \
        --save edsr_x$SCALE/w$3a$2-large
}

# ------------------------------------------------------ ablations (Table 4) ---
edsr_plq_only() {   # piecewise linear quantizer, vanilla finetuning
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --model EDSR --scale $SCALE \
        --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
        --pre_train ../pretrained_model/edsr_baseline_x$SCALE.pt \
        --dir_data $DATA_DIR --data_test Set5 \
        --quantize_a $2 --quantize_w $3 --fq \
        --epochs 10 --batch_size_calib 16 --batch_size_update 2 --num_data 100 --patch_size 384 \
        --quantizer 'minmax' --quantizer_w 'omse' --bp_init 'asym' \
        --lr_w 0.001 --lr_a 0.001 --lr_bp 0.001 \
        --w_sktloss 10.0 --lambda_rec 5.0 --bac \
        --save edsr_x$SCALE/w$3a$2-plq-only --seed 100
}

"$@"
