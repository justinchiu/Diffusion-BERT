#torchrun --nnodes=1 --nproc-per-node=4 --master-port=29400 DDP_main.py \
torchrun --nnodes=1 --nproc-per-node=1 --master-port=29400 DDP_main.py \
  --task_name pg19 \
  --max_length 1024 \
  --lr 5e-5 \
  --batch_size 16 \
  --timestep 'layerwise'

# total batch size should be 128
# switch to 32 * 4 if using 4 gpus
