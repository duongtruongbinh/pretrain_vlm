NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}" \
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}" \
CUDA_VISIBLE_DEVICES=1,2 uv run accelerate launch \
  --num_processes "${NUM_PROCESSES:-2}" \
  --num_machines 1 \
  --mixed_precision "${MIXED_PRECISION:-no}" \
  --main_process_port "${MAIN_PROCESS_PORT:-0}" \
  --dynamo_backend no \
  --multi_gpu \
  train.py