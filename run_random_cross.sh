RESULT_DIR=model_random_cross_new
group=4
degree=5
noise=0.05
num_seeds=20
seed=2

python -m main.training --task=random_cross --num_agents=10 --num_groups $group --collision_penalty=5 --env="FormationTorch-v0" --policy="transformer_edge" --cuda --episode 10000 --result_dir $RESULT_DIR --only_obs_edge --clip_speed --seed $seed --noise_scale $noise;

python -m main.train_program --task=random_cross --num_agents=10 --collision_penalty=5 --env="FormationTorch-v0" --policy="transformer_edge" --num_groups $group --result_dir $RESULT_DIR --deg_loss_wt 0.5 --use_soft_with_prog --num_rules $degree --max_deg $degree;

python -m main.training --task=random_cross --num_agents=10 --collision_penalty=5 --env="FormationTorch-v0" --policy="transformer_edge" --cuda --episode 1000 --num_groups $group --result_dir $RESULT_DIR --retrain_with_prog_attn --use_soft_with_prog;

python -m main.testing --task=random_cross --num_agents=10 --collision_penalty=5 --env="FormationTorch-v0" --policy="transformer_edge" --num_groups $group --result_dir $RESULT_DIR --num_seeds $num_seeds --save_dir 0_tf; 

python -m main.testing --task=random_cross --num_agents=10 --collision_penalty=5 --env="FormationTorch-v0" --policy="transformer_edge" --num_groups $group --result_dir $RESULT_DIR --num_seeds $num_seeds --save_dir 1_hard --hard_attn --hard_attn_deg $degree;  

python -m main.testing --task=random_cross --num_agents=10 --collision_penalty=5 --env="FormationTorch-v0" --policy="transformer_edge" --num_groups $group --result_dir $RESULT_DIR --num_seeds $num_seeds --save_dir 3_prog --prog_attn --use_soft_with_prog; 

python -m main.testing --task=random_cross --num_agents=10 --collision_penalty=5 --env="FormationTorch-v0" --policy="transformer_edge" --num_groups $group --result_dir $RESULT_DIR --num_seeds $num_seeds --save_dir 4_prog_retrained --prog_attn --use_retrained_model --use_soft_with_prog;

python -m main.training --task=random_cross --num_agents=10 --num_groups $group --collision_penalty=5 --env="FormationTorch-v0" --policy="transformer_edge" --cuda --episode 5000 --result_dir "$RESULT_DIR"_disthard --only_obs_edge --clip_speed --dist_based_hard --comm_deg $degree --seed $seed --noise_scale $noise;

python -m main.testing --task=random_cross --num_agents=10 --collision_penalty=5 --env="FormationTorch-v0" --policy="transformer_edge" --num_groups $group --result_dir "$RESULT_DIR"_disthard --num_seeds $num_seeds --save_dir 2_dist;

python -m main.plot_results --result_dir=$RESULT_DIR --env="FormationTorch-v0"
