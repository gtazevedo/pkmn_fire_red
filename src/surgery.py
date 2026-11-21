from __future__ import annotations

import os
import pickle
import shutil
from typing import Optional

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

from src.config import BATTLE_SIZE, CFG, GOAL_SIZE, GRID_SIZE, RAM_SIZE, log


def build_default_env_obs_space() -> spaces.Dict:
    """Espaço de observação padrão do PokemonEnv (antes do VecTransposeImage)."""
    return spaces.Dict({
        "image": spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8),
        "ram": spaces.Box(0.0, 1.0, (RAM_SIZE,), dtype=np.float32),
        "collision_grid": spaces.Box(0.0, 1.0, (GRID_SIZE, GRID_SIZE), dtype=np.float32),
        "visited_grid": spaces.Box(0.0, 1.0, (GRID_SIZE, GRID_SIZE), dtype=np.float32),
        "goal": spaces.Box(0.0, 1.0, (GOAL_SIZE,), dtype=np.float32),
        "battle": spaces.Box(0.0, 1.0, (BATTLE_SIZE,), dtype=np.float32),
    })


def to_policy_obs_space(env_obs_space: spaces.Dict) -> spaces.Dict:
    """Converte Dict space do ambiente para o formato transposto usado pela política SB3."""
    converted: dict[str, spaces.Space] = {}
    for key, subspace in env_obs_space.spaces.items():
        if (
            isinstance(subspace, spaces.Box)
            and len(subspace.shape) == 3
            and subspace.shape[2] in (1, 3, 4)
            and subspace.shape[0] > 4
        ):
            h, w, c = subspace.shape
            converted[key] = spaces.Box(
                low=0,
                high=255,
                shape=(c, h, w),
                dtype=subspace.dtype,
            )
        else:
            converted[key] = subspace
    return spaces.Dict(converted)


def compute_key_feature_ranges(
    obs_space: spaces.Dict,
    cnn_output_dim: int = 256,
) -> tuple[dict[str, tuple[int, int]], int]:
    """
    Calcula o intervalo [start, end) de cada chave do Dict no vetor concatenado
    que entra na LSTM (CombinedExtractor segue a ordem das chaves do Dict).
    """
    ranges: dict[str, tuple[int, int]] = {}
    offset = 0
    for key, subspace in obs_space.spaces.items():
        if is_image_space(subspace, normalized_image=True) or (
            isinstance(subspace, spaces.Box) and len(subspace.shape) == 3
        ):
            dim = cnn_output_dim
        else:
            dim = int(np.prod(subspace.shape))
        ranges[key] = (offset, offset + dim)
        offset += dim
    return ranges, offset


def _expand_lstm_input_tensor(
    old_tensor: torch.Tensor,
    old_ranges: dict[str, tuple[int, int]],
    new_ranges: dict[str, tuple[int, int]],
    new_total_dim: int,
) -> torch.Tensor:
    new_tensor = torch.zeros(
        (old_tensor.shape[0], new_total_dim),
        dtype=old_tensor.dtype,
        device=old_tensor.device,
    )
    for key, (old_start, old_end) in old_ranges.items():
        if key not in new_ranges:
            continue
        new_start, new_end = new_ranges[key]
        width = min(old_end - old_start, new_end - new_start)
        new_tensor[:, new_start : new_start + width] = old_tensor[:, old_start : old_start + width]
    return new_tensor


def _patch_obs_buffer(obs_buf: object, target_space: spaces.Dict, default_n_envs: int) -> None:
    if not isinstance(obs_buf, dict):
        return
    n_envs = default_n_envs
    for arr in obs_buf.values():
        if hasattr(arr, "shape") and len(arr.shape) >= 1:
            n_envs = int(arr.shape[0])
            break
    for key, subspace in target_space.spaces.items():
        if key not in obs_buf:
            obs_buf[key] = np.zeros((n_envs, *subspace.shape), dtype=subspace.dtype)


def _patch_vecnorm_file(
    vecnorm_path: str,
    target_env_obs_space: spaces.Dict,
) -> bool:
    if not os.path.exists(vecnorm_path):
        return False

    bak_path = vecnorm_path + ".bak"
    if not os.path.exists(bak_path):
        shutil.copy2(vecnorm_path, bak_path)

    with open(vecnorm_path, "rb") as f:
        vn = pickle.load(f)

    vn.__dict__["observation_space"] = target_env_obs_space
    if vn.__dict__.get("norm_obs", False) or vn.__dict__.get("norm_obs_keys") is not None:
        vn.__dict__["norm_obs_keys"] = list(target_env_obs_space.spaces.keys())
    _patch_obs_buffer(vn.__dict__.get("old_obs"), target_env_obs_space, vn.__dict__.get("num_envs", 1))
    vn.__dict__["venv"] = None
    if "class_attributes" not in vn.__dict__:
        vn.__dict__["class_attributes"] = {}
    if "returns" not in vn.__dict__:
        vn.__dict__["returns"] = np.zeros(vn.__dict__.get("num_envs", 1), dtype=np.float32)

    tmp_path = vecnorm_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(vn, f)
    os.replace(tmp_path, vecnorm_path)
    return True


def perform_model_surgery(
    model_zip_path: str,
    vecnorm_path: Optional[str] = None,
    stamp_path: Optional[str] = None,
    target_obs_space: Optional[spaces.Dict] = None,
    new_stamp: Optional[str] = None,
) -> dict:
    """
    Executa cirurgia de pesos em um checkpoint RecurrentPPO (MultiInputLstmPolicy)
    para adicionar novas chaves de observação sem perder os pesos pré-treinados.
    """
    if not os.path.exists(model_zip_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {model_zip_path}")

    env_obs_space = target_obs_space if target_obs_space is not None else build_default_env_obs_space()
    policy_obs_space = to_policy_obs_space(env_obs_space)

    data, params, pytorch_vars = load_from_zip_file(model_zip_path, device="cpu")
    old_obs_space = data["observation_space"]
    if not isinstance(old_obs_space, spaces.Dict):
        raise ValueError("Cirurgia suporta apenas observation_space do tipo gymnasium.spaces.Dict")

    policy_state = params["policy"]
    cnn_dim = int(policy_state["mlp_extractor.policy_net.0.weight"].shape[1]) if "mlp_extractor.policy_net.0.weight" in policy_state else 256
    old_ranges, old_total_dim = compute_key_feature_ranges(old_obs_space, cnn_output_dim=cnn_dim)
    new_ranges, new_total_dim = compute_key_feature_ranges(policy_obs_space, cnn_output_dim=cnn_dim)

    actor_w = policy_state["lstm_actor.weight_ih_l0"]
    already_upgraded = (
        actor_w.shape[1] == new_total_dim
        and set(old_obs_space.spaces.keys()) == set(policy_obs_space.spaces.keys())
    )

    if not already_upgraded:
        bak_path = model_zip_path + ".bak"
        if not os.path.exists(bak_path):
            shutil.copy2(model_zip_path, bak_path)

        for key in ("lstm_actor.weight_ih_l0", "lstm_critic.weight_ih_l0"):
            if key in policy_state:
                old_w = policy_state[key]
                policy_state[key] = _expand_lstm_input_tensor(old_w, old_ranges, new_ranges, new_total_dim)

        opt_state = params.get("policy.optimizer", {}).get("state", {})
        for pid, st in opt_state.items():
            if not isinstance(st, dict):
                continue
            for moment_key in ("exp_avg", "exp_avg_sq"):
                tensor = st.get(moment_key)
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.shape[1] == old_total_dim:
                    st[moment_key] = _expand_lstm_input_tensor(tensor, old_ranges, new_ranges, new_total_dim)

        data["observation_space"] = policy_obs_space
        n_envs = int(data.get("n_envs", 1))
        _patch_obs_buffer(data.get("_last_obs"), policy_obs_space, n_envs)
        _patch_obs_buffer(data.get("_last_original_obs"), env_obs_space, n_envs)

        save_to_zip_file(model_zip_path, data=data, params=params, pytorch_variables=pytorch_vars)

    vecnorm_patched = False
    if vecnorm_path is not None:
        vecnorm_patched = _patch_vecnorm_file(vecnorm_path, env_obs_space)

    if stamp_path is not None and new_stamp is not None:
        with open(stamp_path, "w") as f:
            f.write(new_stamp)

    summary = {
        "model_zip": model_zip_path,
        "already_upgraded": already_upgraded,
        "old_total_dim": old_total_dim,
        "new_total_dim": new_total_dim,
        "old_ranges": old_ranges,
        "new_ranges": new_ranges,
        "vecnorm_patched": vecnorm_patched,
    }
    log.info(
        f"[Surgery] {model_zip_path} | upgraded={not already_upgraded} | "
        f"LSTM input {old_total_dim} -> {new_total_dim}"
    )
    return summary


def upgrade_checkpoint_if_needed(
    model_zip_path: str,
    vecnorm_path: Optional[str],
    stamp_path: str,
    target_obs_space: spaces.Dict,
    target_stamp: str,
) -> bool:
    """
    Verifica compatibilidade do checkpoint salvo e aplica cirurgia de pesos
    automaticamente quando novas chaves de observação são introduzidas.
    """
    if not os.path.exists(model_zip_path):
        return False

    data, _, _ = load_from_zip_file(model_zip_path, device="cpu")
    saved_obs_space = data.get("observation_space")
    expected_policy_space = to_policy_obs_space(target_obs_space)

    if isinstance(saved_obs_space, spaces.Dict):
        # Cirurgia só se aplica quando as chaves antigas são subconjunto compatível das novas
        shared_keys = set(saved_obs_space.spaces.keys()) & set(expected_policy_space.spaces.keys())
        shapes_match = all(
            saved_obs_space.spaces[k].shape == expected_policy_space.spaces[k].shape
            for k in shared_keys
        )
        if shapes_match and set(saved_obs_space.spaces.keys()).issubset(set(expected_policy_space.spaces.keys())):
            if set(saved_obs_space.spaces.keys()) != set(expected_policy_space.spaces.keys()):
                perform_model_surgery(
                    model_zip_path=model_zip_path,
                    vecnorm_path=vecnorm_path,
                    stamp_path=stamp_path,
                    target_obs_space=target_obs_space,
                    new_stamp=target_stamp,
                )
                return True
            # Mesmo espaço de observação: apenas atualiza stamp se estiver em v29 -> v30
            if os.path.exists(stamp_path):
                with open(stamp_path) as f:
                    saved_stamp = f.read().strip()
            else:
                saved_stamp = ""
            if saved_stamp == "v29_gen6_holy_grail":
                with open(stamp_path, "w") as f:
                    f.write(target_stamp)
                return True
            return saved_stamp == target_stamp

    return False


if __name__ == "__main__":
    zip_path = CFG.model_path + ".zip"
    vn_path = CFG.model_path + "_vecnorm.pkl"
    st_path = CFG.model_path + ".stamp"
    result = perform_model_surgery(
        model_zip_path=zip_path,
        vecnorm_path=vn_path,
        stamp_path=st_path,
        new_stamp=CFG.config_stamp,
    )
    print(result)
