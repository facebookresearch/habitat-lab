export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# python -u -m habitat_baselines.run   --config-name=imagenav/ddppo_imagenav_distance.yaml habitat_baselines.evaluate=False
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=15244 \
  -m habitat_baselines.run \
    --config-name=imagenav/ddppo_imagenav_distance.yaml habitat_baselines.evaluate=False