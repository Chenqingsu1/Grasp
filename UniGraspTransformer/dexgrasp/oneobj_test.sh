export PYTHONPATH=$PYTHONPATH:/PROJECT/UniGraspTransformer/pytorch_kinematics/src
export CPATH=$CONDA_PREFIX/include
python run_online.py \
    --task StateBasedGrasp \
    --algo ppo \
    --seed 1 \
    --rl_device cuda:0 \
    --num_envs 1 \
    --max_iterations 10000 \
    --config dedicated_policy.yaml \
    --test \
    --test_iteration 1 \
    --object_scale_file train_set_results.yaml \
    --start_line 20 \
    --end_line 21