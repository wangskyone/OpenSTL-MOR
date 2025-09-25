cd /nas_data/LSH/OpenSTL-MOR
CUDA_VISIBLE_DEVICES=1 python tools/train.py -d  kth -c configs/kth/MorLSTM-B.py --ex_name work_dirs/kth/morlstm/vit_dim192 -b 8 -e 100
# python tools/test.py -d  kth -c configs/kth/TAU.py --ex_name work_dirs/kth/tau -b 2 -vb 4 --test
# python tools/test.py -d  kth -c configs/kth/TAU.py --ex_name work_dirs/kth/tau -vb 12 --test


# CUDA_VISIBLE_DEVICES=1 python tools/train.py -d  kth -c configs/kth/simvp/SimVP_ViT.py --ex_name work_dirs/kth/simvp/vit -b 2