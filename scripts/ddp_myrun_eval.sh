# load from pretrained
#torchrun --nnodes=1 --nproc-per-node=4 --master-port=29400 DDP_eval.py \

runeval () {
torchrun --nnodes=1 --nproc-per-node=1 --master-port=29401 DDP_eval.py \
  --lr 5e-5 \
  --batch_size 32 \
  --timestep 'layerwise' \
  --model_name_or_path bert-large-uncased \
  --load_step 74999 \
  --eval_step_size $1 \
  --length_min $2 \
  --length_max $3 \
  --num_batches 10
}

runeval_big () {
torchrun --nnodes=1 --nproc-per-node=1 --master-port=29401 DDP_eval.py \
  --lr 5e-5 \
  --batch_size 32 \
  --timestep 'layerwise' \
  --model_name_or_path bert-large-uncased \
  --load_step 194999 \
  --eval_step_size $1 \
  --length_min $2 \
  --length_max $3 \
  --num_batches 10
}

# Define the step sizes
step_sizes=(4 16 32 128 256 1024)
#step_sizes=(16 32 128 256 1024)

# Define the start,end pairs
start_end_pairs=(
    "0 32"
    "32 64"
    "64 96"
    "96 128"
)

for stepsize in "${step_sizes[@]}"; do
    # Inner loop for start,end pairs
    for pair in "${start_end_pairs[@]}"; do
        # Split the pair into start and end variables
        read -r start end <<< "$pair"
        
        # Build the command string and use eval to execute it
        runeval $stepsize $start $end
        runeval_big $stepsize $start $end
    done
done
