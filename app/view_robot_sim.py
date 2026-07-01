# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot the whole-body retargeting degrees of freedom as a sparse 3D structure.

This intentionally does not show robot meshes, collision shapes, hands, or every
internal joint. It only draws the high-level IK-mapped body structure used by
the retargeter.

Examples:
    uv run python app/view_robot_sim.py --target unitree_h1 --show
    uv run python app/view_robot_sim.py --target unitree_h1 --output h1_dof.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import warp as wp

import newton

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.utils.newton_utils as newton_utils
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
from soma_retargeter.robotics.robot_loader import create_robot_builder
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str


def _target_short_name(target: str) -> str:
    if target.startswith("unitree_"):
        return target.removeprefix("unitree_")
    return target


def _resolve_file(path_text: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if path.exists():
        return path.resolve()

    config_path = io_utils.get_config_file(str(path_text))
    if config_path.exists():
        return config_path.resolve()

    raise FileNotFoundError(f"File not found: {path_text}")


def _default_retargeter_config(target: str) -> pathlib.Path:
    short_name = _target_short_name(target)
    return io_utils.get_config_file(target, f"soma_to_{short_name}_retargeter_config.json")


def _select_scaler_config(
    retargeter_config: dict,
    explicit_scaler_config: str | None,
    prefer_optimized: bool,
) -> pathlib.Path:
    if explicit_scaler_config:
        return _resolve_file(explicit_scaler_config)

    scaler_path = io_utils.get_config_file(retargeter_config["human_robot_scaler_config"])
    if prefer_optimized:
        optimized_path = scaler_path.with_name(f"{scaler_path.stem}_optimized{scaler_path.suffix}")
        if optimized_path.exists():
            return optimized_path.resolve()

    return scaler_path.resolve()


def _mapped_parent(index: int, parents: list[int], visible_mask: np.ndarray) -> int:
    parent = parents[index]
    while parent >= 0 and not visible_mask[parent]:
        parent = parents[parent]
    return parent


def _make_soma_instance(bvh_file: pathlib.Path, facing_direction: str) -> tuple[SkeletonInstance, object]:
    skeleton, animation = bvh_utils.load_bvh(bvh_file)
    root_xform = SpaceConverter(get_facing_direction_type_from_str(facing_direction)).transform(wp.transform_identity())
    skeleton_instance = SkeletonInstance(skeleton, [0.0, 0.0, 0.0], root_xform)
    skeleton_instance.set_local_transforms(animation.get_local_transforms(0))
    return skeleton_instance, skeleton


def _robot_rest_effectors(target: str, retargeter_config: dict, mapped_joints: list[str]) -> np.ndarray:
    robot_builder = create_robot_builder(target, retargeter_config.get("robot_model"))

    builder = newton.ModelBuilder()
    builder.add_builder(robot_builder, wp.transform_identity())
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    body_q = state.body_q.numpy()
    body_names = [newton_utils.get_name_from_label(label) for label in model.body_label]
    ik_map = retargeter_config.get("ik_map", {})

    effectors = np.zeros((len(mapped_joints), 7), dtype=np.float32)
    for i, joint_name in enumerate(mapped_joints):
        mapping = ik_map.get(joint_name)
        if mapping is None:
            continue

        t_body = mapping["t_body"]
        r_body = mapping["r_body"]
        if t_body not in body_names or r_body not in body_names:
            print(f"[WARNING]: Skip {joint_name}: missing body {t_body!r} or {r_body!r}.")
            continue

        effectors[i, 0:3] = body_q[body_names.index(t_body)][0:3]
        effectors[i, 3:7] = body_q[body_names.index(r_body)][3:7]

    return effectors


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    half_range = float(np.max(maxs - mins) * 0.55)
    if half_range < 1e-5:
        half_range = 0.5

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)


def plot_whole_body_dof(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    retargeter_config_file = _resolve_file(args.retargeter_config or _default_retargeter_config(args.target))
    retargeter_config = io_utils.load_json(retargeter_config_file)
    scaler_config_file = _select_scaler_config(
        retargeter_config,
        args.scaler_config,
        prefer_optimized=not args.no_prefer_optimized_scaler,
    )
    bvh_file = _resolve_file(args.bvh)

    skeleton_instance, skeleton = _make_soma_instance(bvh_file, args.facing_direction)
    scaler = HumanToRobotScaler(skeleton, retargeter_config["model_height"], scaler_config_file)

    ik_joint_names = set(retargeter_config.get("ik_map", {}).keys())
    visible_mask = np.array([name in ik_joint_names for name in scaler.mapped_joints], dtype=bool)
    visible_indices = np.flatnonzero(visible_mask)
    if len(visible_indices) == 0:
        raise ValueError("No shared IK-mapped joints found between scaler config and retargeter config.")

    soma_effectors = scaler.compute_effectors_from_skeleton(skeleton_instance, scale_animation=True)
    robot_effectors = _robot_rest_effectors(args.target, retargeter_config, scaler.mapped_joints)

    robot_root = robot_effectors[visible_indices[0], 0:3].copy()
    soma_root = soma_effectors[visible_indices[0], 0:3].copy()
    robot_points = robot_effectors[:, 0:3] - robot_root
    soma_points = soma_effectors[:, 0:3] - soma_root

    mapped_ancestors = [
        _mapped_parent(i, scaler.mapped_joint_parents, visible_mask)
        for i in range(len(scaler.mapped_joints))
    ]

    fig = plt.figure(figsize=(args.figure_size, args.figure_size))
    ax = fig.add_subplot(111, projection="3d")

    for child_idx in visible_indices:
        parent_idx = mapped_ancestors[child_idx]
        if parent_idx < 0:
            continue

        parent = robot_points[parent_idx]
        child = robot_points[child_idx]
        ax.plot([parent[0], child[0]], [parent[1], child[1]], [parent[2], child[2]], "g-", alpha=0.65)

        parent = soma_points[parent_idx]
        child = soma_points[child_idx]
        ax.plot([parent[0], child[0]], [parent[1], child[1]], [parent[2], child[2]], "r-", alpha=0.65)

    visible_robot = robot_points[visible_indices]
    visible_soma = soma_points[visible_indices]
    ax.scatter(
        visible_robot[:, 0],
        visible_robot[:, 1],
        visible_robot[:, 2],
        c="green",
        marker="s",
        s=45,
        label="Robot Target (Rest Pose)",
    )
    ax.scatter(
        visible_soma[:, 0],
        visible_soma[:, 1],
        visible_soma[:, 2],
        c="red",
        marker="o",
        s=45,
        label="SOMA Scaled (T-Pose)",
    )

    if args.labels:
        for idx in visible_indices:
            point = robot_points[idx]
            ax.text(point[0], point[1], point[2], scaler.mapped_joints[idx], fontsize=8, color="green")

    visible_points = np.concatenate([visible_robot, visible_soma], axis=0)
    _set_equal_axes(ax, visible_points)

    ax.set_title(f"Whole-Body DOF Structure ({args.target}, {len(visible_indices)} mapped points)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.legend()
    plt.tight_layout()

    if args.output:
        output_path = pathlib.Path(args.output)
        fig.savefig(output_path, dpi=args.dpi)
        print(f"[INFO]: Saved whole-body DOF plot to: {output_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[INFO]: target: {args.target}")
    print(f"[INFO]: retargeter config: {retargeter_config_file}")
    print(f"[INFO]: scaler config: {scaler_config_file}")
    print(f"[INFO]: shown whole-body mapped points: {len(visible_indices)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the high-level whole-body DOF structure.")
    parser.add_argument("--target", default="unitree_h1", help="Robot target, e.g. unitree_h1 or unitree_g1.")
    parser.add_argument("--retargeter-config", default=None, help="Retargeter config JSON.")
    parser.add_argument("--scaler-config", default=None, help="Scaler config JSON. Defaults to retargeter config.")
    parser.add_argument(
        "--no-prefer-optimized-scaler",
        action="store_true",
        help="Do not auto-use *_optimized.json beside the scaler config.",
    )
    parser.add_argument("--bvh", default="soma/soma_zero_frame0.bvh", help="SOMA rest-pose BVH file.")
    parser.add_argument("--facing-direction", default="Maya", help="Input SOMA facing direction.")
    parser.add_argument("--output", default="whole_body_dof.png", help="Path to save the plot. Use '' to skip saving.")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    parser.add_argument("--labels", action="store_true", help="Label robot mapped points.")
    parser.add_argument("--figure-size", type=float, default=10.0, help="Matplotlib figure size in inches.")
    parser.add_argument("--dpi", type=int, default=140, help="Output image DPI.")
    parser.add_argument("--elev", type=float, default=20.0, help="3D view elevation.")
    parser.add_argument("--azim", type=float, default=-60.0, help="3D view azimuth.")
    args = parser.parse_args()

    if args.output == "":
        args.output = None

    plot_whole_body_dof(args)


if __name__ == "__main__":
    main()
