#torchrun --nnodes=1 --nproc-per-node=2 --master-port=29402 DDP_main.py \
torchrun --nnodes=1 --nproc-per-node=1 --master-port=29401 DDP_main.py \
  --task_name pg19 \
  --max_length 512 \
  --lr 5e-5 \
  --batch_size 32 \
  --timestep 'layerwise' \
  --model_name_or_path mosaicml/mosaic-bert-base-seqlen-2048 \
  --wrap_text \
  --epochs 10 \
  --num_steps 2048 \
  --logging_steps 100
# --batch_size 16 for a100
# --batch_size 4 for rush-compute-01
#
# trying 3072 steps
