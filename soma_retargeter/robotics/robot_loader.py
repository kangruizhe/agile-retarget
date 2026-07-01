# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import newton
import warnings

import soma_retargeter.utils.io_utils as io_utils


_DEFAULT_ROBOT_MODELS = {
    "unitree_g1": {
        "format": "newton_asset_mjcf",
        "asset": "unitree_g1",
        "path": "mjcf/g1_29dof_rev_1_0.xml",
    },
    "unitree_h1": {
        "format": "mjcf",
        "path": "unitree_h1/h1_with_hand.xml",
    },
}


def _find_cached_newton_asset_path(asset_name: str, relative_path: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "newton"
    if not cache_root.exists():
        return None

    matches = sorted(cache_root.glob(f"newton-assets_{asset_name}_*/{asset_name}/{relative_path}"))
    for match in matches:
        if match.is_file():
            return match

    return None


def create_robot_builder(robot_type: str, robot_model_config: dict | None = None) -> newton.ModelBuilder:
    """Create a Newton model builder for a supported robot target."""
    config = robot_model_config or _DEFAULT_ROBOT_MODELS.get(robot_type)
    if config is None:
        raise ValueError(f"[ERROR]: No robot model config registered for [{robot_type}].")

    builder = newton.ModelBuilder()
    model_format = config["format"]

    if model_format == "newton_asset_mjcf":
        asset_path = _find_cached_newton_asset_path(config["asset"], config["path"])
        if asset_path is None:
            asset_path = newton.utils.download_asset(config["asset"]) / config["path"]
        builder.add_mjcf(asset_path)
    elif model_format == "mjcf":
        builder.add_mjcf(str(io_utils.get_config_file(config["path"])))
    elif model_format == "urdf":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*could not resolve package.*")
            builder.add_urdf(
                str(io_utils.get_config_file(config["path"])),
                floating=config.get("floating", True),
                collapse_fixed_joints=config.get("collapse_fixed_joints", False),
                hide_visuals=config.get("hide_visuals", False),
                parse_visuals_as_colliders=config.get("parse_visuals_as_colliders", False),
                enable_self_collisions=config.get("enable_self_collisions", True),
                ignore_inertial_definitions=config.get("ignore_inertial_definitions", False),
            )
    else:
        raise ValueError(f"[ERROR]: Unsupported robot model format [{model_format}].")

    return builder
