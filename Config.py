class PPOConfig:
    total_steps: int = 1_000_000
    num_envs: int = 8
    evaluate_every: int = 10_000
    eval_episodes: int = 10
    
    # rollout buffer
    n_steps: int = 128 # steps collected per env before an update one rollout = n_steps * num_envs transitions
    
    # ppo updates
    n_epochs: int = 4
    minibatch_size: int = 256
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    
    hidden_size: int = 64
    num_rep: int = 5
    smoothing_window: int = 11
    