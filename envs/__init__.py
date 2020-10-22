from gym.envs.registration import register


register(
    id='FormationTorch-v0',
    entry_point='envs.formation_flying:FormationFlyingTorchEnv',
    max_episode_steps=500,
)

register(
    id='UnlabeledGoals-v0',
    entry_point='envs.unlabeled_goals:UnlabeledGoalsEnv',
    max_episode_steps=500,
)

