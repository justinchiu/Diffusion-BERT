torchrun --nnodes=1 --nproc-per-node=1 --master-port=29400 DDP_main.py \
  --lr 5e-5 \
  --batch_size 32 \
  --timestep 'layerwise' \
  --from_scratch false \
  --model_name_or_path bert-large-uncased
