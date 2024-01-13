torchrun --nnodes=1 --nproc-per-node=4 --master-port=29400 DDP_main.py \
  --task_name pg19 \
  --max_length 4096 \
  --lr 5e-5 \
  --batch_size 32 \
  --timestep 'layerwise' \
  --model_name_or_path JunxiongWang/BiGS_512
