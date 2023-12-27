torchrun --nnodes 1 --master-port=29400 DDP_main.py \
  --lr 5e-5 \
  --batch_size 128 \
  --timestep 'layerwise' \
  --from_scratch false
