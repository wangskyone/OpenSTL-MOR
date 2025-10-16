export CUDA_VISIBLE_DEVICES=3
cd /nas_data/LSH/OpenSTL-MOR

# nohup python tools/train.py -d human -c configs/human/PredFormer.py --ex_name work_dirs/human/predformer/facTSattion_onecycle_dim256_head8_ndepth6_patch8_tauloss -b 8 -vb 8 -e 50 --lr 1e-3 > train2.log &
nohup python tools/train.py \
    -d human -c configs/human/ModPredFormer.py \
    --ex_name work_dirs/human/predformer/TSSTattion_onecycle_dim256_head8_ndepth3_patch8_mod0_3_routerloss_flow2_diff1 \
    -b 8 \
    -vb 8 \
    -e 50 \
    --lr 1e-3 > train3.log &