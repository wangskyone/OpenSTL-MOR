export CUDA_VISIBLE_DEVICES=2
cd /nas_data/LSH/OpenSTL-MOR
# CUDA_VISIBLE_DEVICES=1 python tools/train.py -d  kth -c configs/kth/MorLSTM-B.py --ex_name work_dirs/kth/morlstm/vit_dim192 -b 8 -e 100
# python tools/test.py -d  kth -c configs/kth/TAU.py --ex_name work_dirs/kth/tau -b 2 -vb 4 --test
# python tools/test.py -d  kth -c configs/kth/TAU.py --ex_name work_dirs/kth/tau -vb 12 --test


# CUDA_VISIBLE_DEVICES=1 python tools/train.py -d  kth -c configs/kth/simvp/SimVP_ViT.py --ex_name work_dirs/kth/simvp/vit -b 2
# CUDA_VISIBLE_DEVICES=1 python tools/train.py -d  kth -c configs/kth/Patch.py --ex_name work_dirs/kth/patch/modvit_layer -b 8 -e 100
# nohup python tools/train.py -d  kth -c configs/kth/VMRNN-B.py --ex_name work_dirs/kth/vmrnn/vmrnnb_bs4_lr1e-4_dp6_ep100_cosine -b 4 -e 100 > train.log &
# nohup python tools/train.py -d kth -c configs/kth/PredFormer.py --ex_name work_dirs/kth/predformer/facTSattion_onecycle_ndepth4_patch8_drop0_25_epoch200_tauloss -b 2 -vb 2 -e 200 --lr 2.5e-4 > train1.log &
nohup python tools/train.py \
    -d kth \
    -c configs/kth/PredFormer.py \
    --ex_name work_dirs/kth/predformer/TSSTattion_dim256_head8_ndepth2_patch8_onecycle \
    -b 16 \
    -vb 16 \
    --lr 1e-3 -e 100 > train2.log &