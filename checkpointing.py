import getpass
import os
import re
import time
from typing import Dict, Optional

import yaml


def _sanitize_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "run"


def default_checkpoint_root() -> str:
    if os.environ.get("CRAFTAX_CHECKPOINT_ROOT"):
        return os.environ["CRAFTAX_CHECKPOINT_ROOT"]

    if os.environ.get("SCRATCH"):
        return os.path.join(
            os.environ["SCRATCH"], "craftax_baselines", "checkpoints"
        )

    user = getpass.getuser()
    cluster_scratch_root = "/scratch/engin_root/engin1"
    if os.path.exists(cluster_scratch_root):
        return os.path.join(
            cluster_scratch_root, user, "craftax_baselines", "checkpoints"
        )

    return os.path.join("/scratch", user, "craftax_baselines", "checkpoints")


def get_checkpoint_root(config: Optional[Dict] = None) -> str:
    if config is not None and config.get("CHECKPOINT_ROOT"):
        return config["CHECKPOINT_ROOT"]
    return default_checkpoint_root()


def get_checkpoint_run_tag(config: Dict):
    try:
        import wandb
    except ImportError:
        wandb = None

    if wandb is not None and wandb.run is not None:
        return _sanitize_path_component(wandb.run.id)

    env_name = _sanitize_path_component(config.get("ENV_NAME", "craftax"))
    seed = config.get("SEED", "noseed")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{env_name}-seed{seed}-{timestamp}"


def get_checkpoint_dir(config: Dict, dir_name: str = "policies") -> str:
    return os.path.join(
        get_checkpoint_root(config),
        get_checkpoint_run_tag(config),
        dir_name,
    )


def record_checkpoint_reference(checkpoint_dir: str, dir_name: str = "policies") -> None:
    try:
        import wandb
    except ImportError:
        wandb = None

    if wandb is None or wandb.run is None:
        return

    os.makedirs(wandb.run.dir, exist_ok=True)
    pointer_path = os.path.join(wandb.run.dir, f"{dir_name}_path.txt")
    with open(pointer_path, "w") as f:
        f.write(checkpoint_dir + "\n")


def write_config_snapshot(config: Dict, checkpoint_dir: str) -> str:
    run_dir = os.path.dirname(checkpoint_dir.rstrip("/"))
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=True)
    return config_path


def resolve_checkpoint_dir(base_path: str, dir_name: str = "policies") -> str:
    candidate_bases = [base_path]
    candidate_bases.append(os.path.join(base_path, "files"))

    for candidate_base in candidate_bases:
        if os.path.basename(candidate_base.rstrip("/")) == dir_name and os.path.isdir(
            candidate_base
        ):
            return candidate_base

        direct_dir = os.path.join(candidate_base, dir_name)
        if os.path.isdir(direct_dir):
            return direct_dir

        pointer_path = os.path.join(candidate_base, f"{dir_name}_path.txt")
        if os.path.isfile(pointer_path):
            with open(pointer_path) as f:
                resolved_path = f.read().strip()
            if os.path.isdir(resolved_path):
                return resolved_path
            raise ValueError(
                f"Checkpoint pointer exists but target is missing: {resolved_path}"
            )

    raise ValueError(
        f"Could not find {dir_name}/ or {dir_name}_path.txt under {base_path}"
    )
