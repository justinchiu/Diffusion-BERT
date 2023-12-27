torchrun --nproc_per_node 1 --master_port 29501 DDP_main.py \
  --lr 5e-5 \
  --batch_size 128 \
  --timestep 'layerwise' \
  --from_scratch false
