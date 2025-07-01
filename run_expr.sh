export CUDA_VISIBLE_DEVICES=0,1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=15244 \
  -m habitat_baselines.run \
    --config-name=imagenav/ddppo_imagenav_gibson.yaml