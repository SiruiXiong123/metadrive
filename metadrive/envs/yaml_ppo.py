import math

def build_ppo_params(cfg_algo: dict, n_envs: int):
    # 1) 解析 YAML
    lr = cfg_algo.get("learning_rate", 3e-4)
    lr_decay = cfg_algo.get("learning_rate_decay", False)
    cr = cfg_algo.get("clip_range", 0.2)
    cr_decay = cfg_algo.get("clip_range_decay", False)
    cr_vf = cfg_algo.get("clip_range_vf", None)

    rollout_total = cfg_algo.get("batch_size", 2048)     # YAML: 总样本量 ≈ n_steps * n_envs
    sb3_batch = cfg_algo.get("minibatch_size", 64)       # YAML: 每次更新的子批 → SB3 batch_size

    # 2) 推导 n_steps，并确保可整除
    n_steps = max(1, rollout_total // n_envs)
    actual_rollout = n_steps * n_envs
    if actual_rollout != rollout_total:
        print(f"[warn] 调整rollout：{rollout_total}→{actual_rollout} (n_steps={n_steps}, n_envs={n_envs})")

    if actual_rollout % sb3_batch != 0:
        # 调整到最近可整除的 minibatch
        factors = [b for b in range(8, actual_rollout+1) if actual_rollout % b == 0]
        nearest = min(factors, key=lambda x: abs(x - sb3_batch))
        print(f"[warn] minibatch_size {sb3_batch} 不整除 {actual_rollout}，改为 {nearest}")
        sb3_batch = nearest

    # 3) 组装 SB3 参数
    ppo_params = dict(
        learning_rate = 3e-4,
        n_steps       = n_steps,
        batch_size    = sb3_batch,
        n_epochs      = cfg_algo.get("n_epochs", 10),
        gamma         = cfg_algo.get("gamma", 0.99),
        gae_lambda    = cfg_algo.get("gae_lambda", 0.95),
        clip_range    = 0.2,
        clip_range_vf = cr_vf,
        ent_coef      = cfg_algo.get("ent_coef", 0.01),
        vf_coef       = cfg_algo.get("vf_coef", 0.5),
        max_grad_norm = cfg_algo.get("max_grad_norm", 0.5),
        verbose       = 1,
    )
    return ppo_params
