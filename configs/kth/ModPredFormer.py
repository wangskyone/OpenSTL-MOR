method = 'modpredformer'

model_config = {
    # image h w c
    'height': 128,
    'width': 128,
    'num_channels': 1,
    # video length in and out
    'pre_seq': 10,
    'after_seq': 20,
    # patch size
    'patch_size': 2,
    'dim': 128, 
    'heads': 4,
    'dim_head': 32,
    # dropout
    'dropout': 0.1,
    'attn_dropout': 0.1,
    'drop_path': 0.1,
    'scale_dim': 4,
    # depth
    'depth': 1,
    'Ndepth': 8, # For FullAttention-24, for BinaryST, BinaryST, FacST, FacTS-12, for TST,STS-8, for TSST, STTS-6
    'capacity_factor': 0.1

}

alpha = 0.1 # tau_loss weight
beta = 0.1  # router_loss weight

sched = 'onecycle'