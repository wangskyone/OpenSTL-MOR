cd /nas_data/LSH/OpenSTL-MOR
CUDA_VISIBLE_DEVICES=1 python tools/train.py -d  kth -c configs/kth/MorLSTM-B.py --ex_name work_dirs/kth/morlstm/vit -b 12 -e 100
# python tools/test.py -d  kth -c configs/kth/TAU.py --ex_name work_dirs/kth/tau -b 2 -vb 4 --test