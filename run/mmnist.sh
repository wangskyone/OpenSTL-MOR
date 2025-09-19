cd /nas_data/LSH/OpenSTL
CUDA_VISIBLE_DEVICES=2 python tools/train.py -d  mmnist -c configs/mmnist/SwinLSTM-D.py --ex_name work_dirs/mmnist/swinlstmD 
# python tools/test.py -d  kth -c configs/kth/TAU.py --ex_name work_dirs/kth/tau -b 2 -vb 4 --test