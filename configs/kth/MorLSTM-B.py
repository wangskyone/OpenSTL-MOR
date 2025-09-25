method = 'morlstm_b'
# model
# depths_downsample = '2,6'
# depths_upsample = '6,2'
# num_heads = '4,8'
# patch_size = 2
# window_size = 4
# embed_dim = 128
patch_size = 4 
embed_dim =192
depths =6
num_heads = 8

# training
lr = 1e-5
batch_size = 16
sched = 'cosine'
debug = False