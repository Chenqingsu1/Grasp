export PYTHONPATH=$PYTHONPATH:/PROJECT/UniGraspTransformer/pytorch_kinematics/src
export CPATH=$CONDA_PREFIX/include
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
# export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# export VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d
# python run_online.py \
#     --task StateBasedGrasp \
#     --algo ppo \
#     --seed 1 \
#     --rl_device cuda:0 \
#     --num_envs 1\
#     --max_iterations 10000 \
#     --config dedicated_policy.yaml \
#     --headless \
#     --object_scale_file train_set_results.yaml \
#     --start_line 0 \
#     --end_line 1 

python run_online.py \
    --task StateBasedGrasp \
    --algo ppo \
    --seed 0 \
    --rl_device cuda:0 \
    --num_envs 1\
    --max_iterations 10000 \
    --config dedicated_policy_xhandarm.yaml \
    --object_scale_file train_set_results.yaml \
    --start_line 28\
    --end_line 29