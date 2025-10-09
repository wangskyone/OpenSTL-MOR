method = 'patch'
# model

model_type = 'modvit'
hid_S = 64
hid_T = 256
N_T = 6

# training
lr = 1e-3
drop_path = 0.1
batch_size = 16  # bs = 2 x 4GPUs
sched = 'onecycle'