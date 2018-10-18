#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6,7 python main.py --data-dir ~/Code/DeepCC_one_hop/experiments/1fps_L1L2/L1-tracklets/hyperEMB_1.h5 --save_result
CUDA_VISIBLE_DEVICES=6,7 python main.py --data-dir ~/Code/DeepCC_one_hop/experiments/1fps_L1L2/L1-tracklets/hyperEMB_2.h5 --save_result
CUDA_VISIBLE_DEVICES=6,7 python main.py --data-dir ~/Code/DeepCC_one_hop/experiments/1fps_L1L2/L1-tracklets/hyperEMB_3.h5 --save_result
CUDA_VISIBLE_DEVICES=6,7 python main.py --data-dir ~/Code/DeepCC_one_hop/experiments/1fps_L1L2/L1-tracklets/hyperEMB_4.h5 --save_result
CUDA_VISIBLE_DEVICES=6,7 python main.py --data-dir ~/Code/DeepCC_one_hop/experiments/1fps_L1L2/L1-tracklets/hyperEMB_5.h5 --save_result
CUDA_VISIBLE_DEVICES=6,7 python main.py --data-dir ~/Code/DeepCC_one_hop/experiments/1fps_L1L2/L1-tracklets/hyperEMB_6.h5 --save_result
CUDA_VISIBLE_DEVICES=6,7 python main.py --data-dir ~/Code/DeepCC_one_hop/experiments/1fps_L1L2/L1-tracklets/hyperEMB_7.h5 --save_result
CUDA_VISIBLE_DEVICES=6,7 python main.py --data-dir ~/Code/DeepCC_one_hop/experiments/1fps_L1L2/L1-tracklets/hyperEMB_8.h5 --save_result
