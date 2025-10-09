method = 'VMRNN_B'
# model
patch_size = 2
depths = 6
window_size = 4
drop_rate = 0.
attn_drop_rate = 0.
drop_path_rate = 0.1
# 
embed_dim = 128
num_heads = 8
# training
batch_size = 8
lr = 1e-4
sched = 'cosine'
warmup_epoch = 5