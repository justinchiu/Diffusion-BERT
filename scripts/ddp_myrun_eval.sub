# load from pretrained
#torchrun --nnodes=1 --nproc-per-node=4 --master-port=29400 DDP_eval.py \
torchrun --nnodes=1 --nproc-per-node=1 --master-port=29401 DDP_eval.py \
  --lr 5e-5 \
  --batch_size 32 \
  --timestep 'layerwise' \
  --model_name_or_path bert-base-uncased \
  --load_step 119999 \
  --eval_step_size 1024

