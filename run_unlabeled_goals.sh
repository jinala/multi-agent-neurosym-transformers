RESULT_DIR=model_unlabeled_new
degree=5
num_seeds=20
seed=2
env="UnlabeledGoals-v0"


python -m main.training --task=unlabeled --num_agents=10 --collision_penalty=0 --env "$env" --policy="transformer_edge" --cuda --episode 10000 --result_dir $RESULT_DIR --only_obs_edge --clip_speed --detach_state --hops=2 --max_step=100 --seed=$seed

python -m main.testing --task=unlabeled --num_agents=10 --collision_penalty=0 --env=$env --policy="transformer_edge" --result_dir $RESULT_DIR  --num_seeds $num_seeds --save_dir 0_tf --max_step=100 ;

python -m main.testing --task=unlabeled --num_agents=10 --collision_penalty=0 --env=$env --policy="transformer_edge" --result_dir $RESULT_DIR  --num_seeds $num_seeds --save_dir 1_hard --hard_attn --max_step=100 --hard_attn_deg $degree;


python -m main.train_program --task=unlabeled --num_agents=10 --collision_penalty=0 --env=$env --policy="transformer_edge" --result_dir $RESULT_DIR --deg_loss_wt 0.7 --use_soft_with_prog --num_rules $degree --max_deg $degree --max_step=100 --sep_hops

python -m main.testing --task=unlabeled --num_agents=10 --collision_penalty=0 --env=$env --policy="transformer_edge" --result_dir $RESULT_DIR --num_seeds $num_seeds --save_dir 3_prog --prog_attn --use_soft_with_prog --max_step=100 --sep_hops

python -m main.training --task=unlabeled --num_agents=10 --collision_penalty=0 --env=$env --policy="transformer_edge" --cuda --episode 1000 --result_dir $RESULT_DIR --retrain_with_prog_attn --use_soft_with_prog --max_step=100 --detach_state --sep_hops

python -m main.testing --task=unlabeled --num_agents=10 --collision_penalty=0 --env=$env --policy="transformer_edge" --result_dir $RESULT_DIR --num_seeds $num_seeds --save_dir 4_prog_retrained --prog_attn --use_retrained_model --use_soft_with_prog --max_step=100 --sep_hops




python -m main.training --task=unlabeled --num_agents=10 --collision_penalty=0 --env=$env --policy="transformer_edge" --cuda --episode 10000 --result_dir "$RESULT_DIR"_disthard --only_obs_edge --clip_speed --seed=$seed --detach_state --hops=2 --max_step=100 --dist_based_hard --comm_deg=$degree 

python -m main.testing --task=unlabeled --num_agents=10 --collision_penalty=0 --env=$env --policy="transformer_edge" --result_dir "$RESULT_DIR"_disthard  --num_seeds $num_seeds --save_dir 2_dist --max_step=100 ;


python -m main.plot_results --result_dir=$RESULT_DIR --env="$env"

