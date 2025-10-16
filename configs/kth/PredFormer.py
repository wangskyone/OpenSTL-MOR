method = 'PredFormer'

model_config = {
    # image h w c
    'height': 128,
    'width': 128,
    'num_channels': 1,
    # video length in and out
    'pre_seq': 10,
    'after_seq': 20,
    # patch size
    'patch_size': 8,
    'dim': 256, 
    'heads': 8,
    'dim_head': 32,
    # dropout
    'dropout': 0.1,
    'attn_dropout': 0.1,
    'drop_path': 0.25,
    'scale_dim': 2,
    # depth
    'depth': 1,
    'Ndepth': 2 

}
alpha = 0.1 # tau_loss weight
sched = 'onecycle'