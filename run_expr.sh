export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=1  
export OPENBLAS_NUM_THREADS=1 
export MKL_NUM_THREADS=1   
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 
export TORCH_NUM_THREADS=1 

# python -u -m habitat_baselines.run --config-name=imagenav/ddppo_image.yaml habitat_baselines.evaluate=False
# python -u -m habitat_baselines.run --config-name=imagenav/ddppo_imagenav_distance.yaml habitat_baselines.evaluate=False
# python -u -m habitat_baselines.run --config-name=imagenav/ddppo_distance_gt.yaml habitat_baselines.evaluate=False
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=15244 \
  -m habitat_baselines.run \
    --config-name=imagenav/ddppo_image.yaml habitat_baselines.evaluate=False