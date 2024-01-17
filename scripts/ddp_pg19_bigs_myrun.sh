#torchrun --nnodes=1 --nproc-per-node=4 --master-port=29400 DDP_main.py \
torchrun --nnodes=1 --nproc-per-node=1 --master-port=29400 DDP_main.py \
  --task_name pg19 \
  --max_length 4094 \
  --lr 5e-5 \
  --batch_size 1 \
  --timestep 'layerwise' \
  --from_scratch True \
  --model_name_or_path JunxiongWang/BiGS_4096

# total batch size should be 128
# switch to 32 * 4 if using 4 gpus
