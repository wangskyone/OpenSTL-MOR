export CUDA_VISIBLE_DEVICES=2
cd /nas_data/LSH/OpenSTL-MOR

nohup python tools/train.py -d human -c configs/human/PredFormer.py --ex_name work_dirs/human/predformer/facTSattion_onecycle_dim256_head8_ndepth6_patch8_tauloss -b 8 -vb 8 -e 50 --lr 1e-3 > train2.log &