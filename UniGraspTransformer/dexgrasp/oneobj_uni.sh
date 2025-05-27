export PYTHONPATH=$PYTHONPATH:/PROJECT/UniGraspTransformer/pytorch_kinematics/src
export CPATH=$CONDA_PREFIX/include
python run_online.py \
    --task StateBasedGrasp \
    --algo dagger_value \
    --seed 0 \
    --rl_device cuda:0 \
    --num_envs 1000 \
    --max_iterations 10000 \
    --config universal_policy_state_based.yaml \
    --headless \
    --test \
    --test_iteration 1 \
    --model_dir distill_0000_0009 \
    --object_scale_file train_set_results.yaml \
    --start_line 0 \
    --end_line 1