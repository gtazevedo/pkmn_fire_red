import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.config import CFG, GOAL_SIZE, GRID_SIZE, RAM_SIZE
from src.surgery import perform_model_surgery, upgrade_checkpoint_if_needed


class _DummyEnv(gym.Env):
    def __init__(self, obs_space: spaces.Dict):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = spaces.Discrete(7)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = {k: np.zeros(s.shape, dtype=s.dtype) for k, s in self.observation_space.spaces.items()}
        return obs, {}

    def step(self, action):
        obs = {k: np.zeros(s.shape, dtype=s.dtype) for k, s in self.observation_space.spaces.items()}
        return obs, 1.0, False, False, {}


def test_weight_surgery_preserves_weights_and_outputs(tmp_path):
    old_space = spaces.Dict({
        "image": spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8),
        "ram": spaces.Box(0.0, 1.0, (RAM_SIZE,), dtype=np.float32),
    })
    new_space = spaces.Dict({
        "image": spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8),
        "ram": spaces.Box(0.0, 1.0, (RAM_SIZE,), dtype=np.float32),
        "collision_grid": spaces.Box(0.0, 1.0, (GRID_SIZE, GRID_SIZE), dtype=np.float32),
        "goal": spaces.Box(0.0, 1.0, (GOAL_SIZE,), dtype=np.float32),
    })

    old_venv = VecNormalize(
        DummyVecEnv([lambda: _DummyEnv(old_space)]),
        norm_obs=False,
        norm_reward=True,
    )
    old_model = RecurrentPPO(
        "MultiInputLstmPolicy",
        old_venv,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        verbose=0,
        seed=123,
        device="cpu",
    )
    old_model.learn(total_timesteps=16)

    model_base = tmp_path / "pokemon_lstm_model"
    zip_path = str(model_base) + ".zip"
    vn_path = str(model_base) + "_vecnorm.pkl"
    stamp_path = str(model_base) + ".stamp"

    old_model.save(str(model_base))
    old_venv.save(vn_path)
    with open(stamp_path, "w") as f:
        f.write("v29_gen6_holy_grail")

    # Avalia distribuição da política antiga numa observação fixa
    rng = np.random.default_rng(0)
    fixed_img = rng.integers(0, 255, size=(1, 84, 84, 1), dtype=np.uint8)
    fixed_ram = rng.random(size=(1, RAM_SIZE), dtype=np.float32)

    old_obs_for_policy = {
        "image": np.transpose(fixed_img, (0, 3, 1, 2)),
        "ram": fixed_ram,
    }
    old_lstm_states = (
        torch.zeros((1, 1, 256)),
        torch.zeros((1, 1, 256)),
    )
    ep_start = torch.ones((1,), dtype=torch.float32)
    with torch.no_grad():
        obs_t, _ = old_model.policy.obs_to_tensor(old_obs_for_policy)
        dist_old, _ = old_model.policy.get_distribution(obs_t, old_lstm_states, ep_start)
        probs_old = dist_old.distribution.probs.clone()

    old_actor_ih = old_model.policy.lstm_actor.weight_ih_l0.detach().clone()
    old_critic_ih = old_model.policy.lstm_critic.weight_ih_l0.detach().clone()

    # Executa cirurgia automática via upgrade_checkpoint_if_needed
    upgraded = upgrade_checkpoint_if_needed(
        model_zip_path=zip_path,
        vecnorm_path=vn_path,
        stamp_path=stamp_path,
        target_obs_space=new_space,
        target_stamp=CFG.config_stamp,
    )
    assert upgraded is True
    assert open(stamp_path).read().strip() == CFG.config_stamp

    # Idempotência: segunda chamada não reescreve dimensões
    summary2 = perform_model_surgery(zip_path, vn_path, stamp_path, target_obs_space=new_space)
    assert summary2["already_upgraded"] is True

    # Carrega o checkpoint cirúrgico no novo ambiente
    new_venv = DummyVecEnv([lambda: _DummyEnv(new_space)])
    new_venv = VecNormalize.load(vn_path, new_venv)
    new_model = RecurrentPPO.load(str(model_base), env=new_venv, device="cpu")

    new_actor_ih = new_model.policy.lstm_actor.weight_ih_l0.detach()
    new_critic_ih = new_model.policy.lstm_critic.weight_ih_l0.detach()
    assert new_actor_ih.shape == (1024, 366)
    assert new_critic_ih.shape == (1024, 366)

    # Colunas antigas preservadas e novas zeradas
    assert torch.all(new_actor_ih[:, 0:87] == 0)
    assert torch.equal(new_actor_ih[:, 87:343], old_actor_ih[:, 0:256])
    assert torch.equal(new_actor_ih[:, 343:366], old_actor_ih[:, 256:279])

    assert torch.all(new_critic_ih[:, 0:87] == 0)
    assert torch.equal(new_critic_ih[:, 87:343], old_critic_ih[:, 0:256])
    assert torch.equal(new_critic_ih[:, 343:366], old_critic_ih[:, 256:279])

    # Verifica estado do otimizador expandido
    _, params, _ = load_from_zip_file(zip_path, device="cpu")
    opt_shapes = [
        st["exp_avg"].shape
        for st in params["policy.optimizer"]["state"].values()
        if isinstance(st.get("exp_avg"), torch.Tensor) and st["exp_avg"].ndim == 2 and st["exp_avg"].shape[0] == 1024
    ]
    assert (1024, 366) in opt_shapes

    # Com collision_grid=0 e goal=0, a distribuição de ações é idêntica à original!
    new_obs_for_policy = {
        "image": np.transpose(fixed_img, (0, 3, 1, 2)),
        "ram": fixed_ram,
        "collision_grid": np.zeros((1, GRID_SIZE, GRID_SIZE), dtype=np.float32),
        "goal": np.zeros((1, GOAL_SIZE), dtype=np.float32),
    }
    with torch.no_grad():
        obs_t_new, _ = new_model.policy.obs_to_tensor(new_obs_for_policy)
        dist_new, _ = new_model.policy.get_distribution(obs_t_new, old_lstm_states, ep_start)
        probs_new = dist_new.distribution.probs.clone()

    assert torch.allclose(probs_old, probs_new, atol=1e-6)

    # Verifica que continuar o treinamento funciona sem erros
    new_model.learn(total_timesteps=16, reset_num_timesteps=False)
    assert new_model.num_timesteps == 32
    new_venv.close()
    old_venv.close()
