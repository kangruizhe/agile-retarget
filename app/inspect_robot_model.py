# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET

import newton

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.robotics.robot_loader import create_robot_builder
from soma_retargeter.utils.newton_utils import get_name_from_label


def _movable_joint_names_from_urdf(path):
    root = ET.parse(path).getroot()
    return [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") not in ("fixed", None)
    ]


def main():
    parser = argparse.ArgumentParser(description="Inspect a Newton robot model target.")
    parser.add_argument("--target", default="unitree_h1", help="Registered robot target name.")
    parser.add_argument("--urdf", default=None, help="Optional config-relative URDF path.")
    args = parser.parse_args()

    if args.urdf:
        builder = newton.ModelBuilder()
        path = io_utils.get_config_file(args.urdf)
        builder.add_urdf(str(path), floating=True, collapse_fixed_joints=False)
        movable_joint_names = _movable_joint_names_from_urdf(path)
    else:
        builder = create_robot_builder(args.target)
        movable_joint_names = []
        if args.target == "unitree_h1":
            movable_joint_names = _movable_joint_names_from_urdf(
                io_utils.get_config_file("unitree_h1", "h1_with_hand.urdf"))

    print(f"target: {args.target}")
    print(f"body_count: {builder.body_count}")
    print(f"joint_count: {builder.joint_count}")
    print(f"joint_coord_count: {builder.joint_coord_count}")
    print(f"joint_dof_count: {builder.joint_dof_count}")
    print("body_names:")
    for body_name in [get_name_from_label(label) for label in builder.body_label]:
        print(f"  {body_name}")
    print("newton_joint_names:")
    for joint_name in [get_name_from_label(label) for label in builder.joint_label]:
        print(f"  {joint_name}")
    if movable_joint_names:
        print("movable_urdf_joint_names:")
        for joint_name in movable_joint_names:
            print(f"  {joint_name}")


if __name__ == "__main__":
    main()
