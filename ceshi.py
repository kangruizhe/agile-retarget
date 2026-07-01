#!/usr/bin/env python3
"""根据机器人配置文件夹打印关节自由度范围。

用法示例：
    python ceshi.py soma_retargeter/configs/unitree_h1
    python ceshi.py soma_retargeter/configs/unitree_g1

脚本会优先读取配置文件夹里的 retargeter 配置。如果配置里有
robot_model.path，就按这个模型文件解析；否则自动在文件夹中查找
.xml/.urdf 模型。G1 这种模型在 Newton 缓存里的目标，会按文件夹名
unitree_g1 自动去缓存里找。

关节范围通常是弧度；脚本会同时打印弧度和角度。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = REPO_ROOT / "soma_retargeter/configs"
G1_CACHE_GLOB = ".cache/newton/newton-assets_unitree_g1_*/unitree_g1/mjcf/g1_29dof_rev_1_0.xml"
MODEL_SUFFIXES = (".xml", ".urdf")


def find_g1_model() -> Path | None:
    matches = sorted(Path.home().glob(G1_CACHE_GLOB))
    return matches[0] if matches else None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_model_path(model_path_text: str, config_dir: Path) -> Path:
    model_path = Path(model_path_text)
    if model_path.is_absolute():
        return model_path

    candidates = [
        REPO_ROOT / model_path,
        CONFIG_ROOT / model_path,
        config_dir / model_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (CONFIG_ROOT / model_path).resolve()


def find_retargeter_config(config_dir: Path) -> Path | None:
    matches = sorted(config_dir.glob("*retargeter_config.json"))
    return matches[0] if matches else None


def find_model_from_config_dir(config_dir: Path) -> Path:
    config_dir = config_dir.resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"配置文件夹不存在: {config_dir}")

    retargeter_config = find_retargeter_config(config_dir)
    if retargeter_config is not None:
        data = load_json(retargeter_config)
        robot_model = data.get("robot_model")
        if isinstance(robot_model, dict) and robot_model.get("path"):
            model_path = resolve_config_model_path(robot_model["path"], config_dir)
            if model_path.exists():
                return model_path
            raise FileNotFoundError(
                f"retargeter 配置指向的模型不存在: {model_path}\n"
                f"配置文件: {retargeter_config}"
            )

    model_files = [
        path
        for path in sorted(config_dir.iterdir())
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    ]
    if model_files:
        xml_files = [path for path in model_files if path.suffix.lower() == ".xml"]
        return (xml_files[0] if xml_files else model_files[0]).resolve()

    if config_dir.name == "unitree_g1":
        g1_model = find_g1_model()
        if g1_model is not None:
            return g1_model.resolve()

    raise FileNotFoundError(
        f"没有在配置文件夹里找到模型文件: {config_dir}\n"
        "需要 .xml/.urdf，或者 retargeter_config.json 里有 robot_model.path。"
    )


def joint_axis_from_urdf(joint: ET.Element) -> str:
    axis = joint.find("axis")
    if axis is not None:
        return axis.attrib.get("xyz", "")
    return ""


def joint_range_from_urdf(joint: ET.Element) -> str | None:
    limit = joint.find("limit")
    if limit is None:
        return None

    lower = limit.attrib.get("lower")
    upper = limit.attrib.get("upper")
    if lower is None or upper is None:
        return None

    return f"{lower} {upper}"


def parse_joint_ranges(model_path: Path) -> list[dict[str, str | float | None]]:
    root = ET.parse(model_path).getroot()
    rows = []
    is_urdf = root.tag == "robot" or model_path.suffix.lower() == ".urdf"

    for joint in root.iter("joint"):
        name = joint.attrib.get("name", "")
        axis = joint_axis_from_urdf(joint) if is_urdf else joint.attrib.get("axis", "")
        joint_type = joint.attrib.get("type", "")
        range_text = joint_range_from_urdf(joint) if is_urdf else joint.attrib.get("range")

        row: dict[str, str | float | None] = {
            "name": name,
            "type": joint_type,
            "axis": axis,
            "range": range_text,
            "lo_rad": None,
            "hi_rad": None,
            "lo_deg": None,
            "hi_deg": None,
        }

        if range_text:
            parts = range_text.split()
            if len(parts) == 2:
                lo_rad, hi_rad = map(float, parts)
                row.update(
                    {
                        "lo_rad": lo_rad,
                        "hi_rad": hi_rad,
                        "lo_deg": math.degrees(lo_rad),
                        "hi_deg": math.degrees(hi_rad),
                    }
                )

        rows.append(row)

    return rows


def should_show_joint(name: str, only_main_body: bool) -> bool:
    if not only_main_body:
        return True

    keywords = (
        "hip",
        "knee",
        "ankle",
        "waist",
        "torso",
        "shoulder",
        "elbow",
        "wrist",
        "hand",
    )
    return any(keyword in name for keyword in keywords)


def print_model(model_name: str, model_path: Path, only_main_body: bool) -> None:
    rows = parse_joint_ranges(model_path)
    shown_rows = [row for row in rows if should_show_joint(str(row["name"]), only_main_body)]
    ranged_rows = [row for row in shown_rows if row["range"]]

    print(f"\n===== {model_name} =====")
    print(f"model: {model_path}")
    print(f"format: {model_path.suffix.lower().lstrip('.')}")
    print(f"shown joints: {len(shown_rows)}")
    print(f"joints with range: {len(ranged_rows)}")
    print()

    header = (
        f"{'joint':34s} {'axis':9s} "
        f"{'rad min':>10s} {'rad max':>10s} "
        f"{'deg min':>10s} {'deg max':>10s}"
    )
    print(header)
    print("-" * len(header))

    for row in shown_rows:
        name = str(row["name"])
        axis = str(row["axis"] or "-")
        if row["range"]:
            print(
                f"{name:34s} {axis:9s} "
                f"{float(row['lo_rad']):10.4f} {float(row['hi_rad']):10.4f} "
                f"{float(row['lo_deg']):10.1f} {float(row['hi_deg']):10.1f}"
            )
        else:
            joint_type = str(row["type"] or "no-range")
            print(f"{name:34s} {axis:9s} {joint_type:>43s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="根据机器人配置文件夹查看关节自由度范围。")
    parser.add_argument(
        "config_dir",
        type=Path,
        help="机器人配置文件夹，例如 soma_retargeter/configs/unitree_h1。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="显示所有 joint，包括手指等；默认只显示躯干、腿、手臂主链。",
    )
    args = parser.parse_args()

    only_main_body = not args.all
    config_dir = args.config_dir.resolve()
    model_path = find_model_from_config_dir(config_dir)
    print_model(config_dir.name, model_path, only_main_body)


if __name__ == "__main__":
    main()
