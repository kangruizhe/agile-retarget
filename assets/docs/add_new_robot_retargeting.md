# 添加新的机器人重定向目标

这份文档说明如何给 SOMA Retargeter 接入一个新的机器人目标。目标是把 SOMA BVH 动作通过 Newton IK pipeline 重定向成机器人可播放的 CSV 关节数据。

## 总览

接入一个新机器人，通常需要补齐这些东西：

1. 在 `soma_retargeter/configs/<robot_name>/` 下放机器人模型。
2. 在 `soma_retargeter/robotics/robot_loader.py` 里注册模型加载方式。
3. 在 `soma_retargeter/pipelines/utils.py` 里注册 target 类型和字符串。
4. 写 retargeter config，用来描述 SOMA 关节到机器人 link 的 IK 映射。
5. 写 scaler config，用来描述 SOMA 到机器人比例、offset 和姿态对齐。
6. 如果是双足机器人，通常还需要 feet stabilizer config。
7. 在 `assets/` 下写 viewer/batch app config。
8. 如果默认 URDF 关节顺序 CSV 不够用，需要补 CSV 输出格式。

建议一开始不要一次性接全身。先只映射 `Hips`、`Chest`、左右脚；跑通之后再加腿、手臂和手。

## 1. 添加机器人模型文件

创建一个机器人专属配置目录：

```text
soma_retargeter/configs/my_robot/
  my_robot.urdf
  my_robot.xml
  meshes/
```

URDF 和 MJCF 都可以。模型如果引用 mesh，要确保模型文件能解析到对应路径。URDF 里如果有 `package://...`，需要确认 Newton 能解析，或者相关 warning 对加载没有影响。

## 2. 注册机器人加载器

编辑：

```text
soma_retargeter/robotics/robot_loader.py
```

在 `_DEFAULT_ROBOT_MODELS` 里加一项：

```python
"my_robot": {
    "format": "urdf",
    "path": "my_robot/my_robot.urdf",
    "floating": True,
    "collapse_fixed_joints": False,
    "hide_visuals": True,
    "parse_visuals_as_colliders": False,
}
```

常见写法：

```python
{
    "format": "urdf",
    "path": "my_robot/my_robot.urdf",
    "floating": True,
}
```

```python
{
    "format": "mjcf",
    "path": "my_robot/my_robot.xml",
}
```

然后先测试模型能不能加载：

```bash
python - <<'PY'
from soma_retargeter.robotics.robot_loader import create_robot_builder

builder = create_robot_builder("my_robot")
model = builder.finalize()
print("body_count", builder.body_count)
print("joint_coord_count", model.joint_coord_count)
PY
```

## 3. 注册 Target 类型

编辑：

```text
soma_retargeter/pipelines/utils.py
```

添加新的 enum：

```python
class TargetType(IntEnum):
    UNITREE_G1 = auto()
    UNITREE_H1 = auto()
    MY_ROBOT = auto()
```

添加字符串映射：

```python
_TARGET_TYPE_TO_STR = {
    TargetType.UNITREE_G1: "unitree_g1",
    TargetType.UNITREE_H1: "unitree_h1",
    TargetType.MY_ROBOT: "my_robot",
}
```

然后在 `get_retargeter_config()` 里返回新机器人的配置：

```python
elif target == TargetType.MY_ROBOT:
    config_dir = "my_robot"
    filename = "soma_to_my_robot_retargeter_config.json"
```

还要确认 `NewtonPipeline` 接受这个 target。当前代码里 G1/H1 都是通过 `create_robot_builder()` 走统一加载逻辑，新机器人也应该接进同一条分支。

## 4. 添加 Retargeter Config

创建：

```text
soma_retargeter/configs/my_robot/soma_to_my_robot_retargeter_config.json
```

示例：

```json
{
    "robot_model": {
        "format": "urdf",
        "path": "my_robot/my_robot.urdf",
        "floating": true,
        "collapse_fixed_joints": false,
        "hide_visuals": true,
        "parse_visuals_as_colliders": false
    },
    "initialization_pose": "soma/soma_zero_frame0.bvh",
    "num_initialization_frames": 10,
    "num_stabilization_frames": 5,
    "human_robot_scaler_config": "my_robot/soma_to_my_robot_scaler_config_rest_aligned.json",
    "feet_stabilizer_config": "",
    "model_height": 1.70,

    "ik_iterations": 24,
    "joint_limit_weight": 10.0,
    "smooth_joint_filter_weight": 5.5,
    "collision_weight": 0.0,
    "enable_post_processing": false,
    "output_smoothing_window": 1,

    "smooth_joint_filter_objective_body_masks": {},

    "ik_map": {
        "Hips": {
            "t_body": "pelvis",
            "r_body": "pelvis",
            "t_weight": 30.0,
            "r_weight": 2.0
        },
        "Chest": {
            "t_body": "torso_link",
            "r_body": "torso_link",
            "t_weight": 0.7,
            "r_weight": 0.7
        },
        "LeftFoot": {
            "t_body": "left_ankle_link",
            "r_body": "left_ankle_link",
            "t_weight": 30.0,
            "r_weight": 2.0
        },
        "RightFoot": {
            "t_body": "right_ankle_link",
            "r_body": "right_ankle_link",
            "t_weight": 30.0,
            "r_weight": 2.0
        }
    }
}
```

关键字段说明：

- `robot_model`：IK solver 使用的机器人模型。
- `human_robot_scaler_config`：生成 IK target 时使用的 scaler config。
- `ik_map`：SOMA 关节到机器人 body/link 的映射。
- `t_weight`：位置目标权重。
- `r_weight`：旋转目标权重。
- `feet_stabilizer_config`：可选的足部稳定器配置。
- `output_smoothing_window`：离线输出平滑窗口。设为 `1` 表示不平滑。

`ik_map` 里的 `t_body`/`r_body` 必须是 Newton 模型里的 body 名。可以用下面的工具检查：

```bash
python app/inspect_robot_model.py --target my_robot
```

## 5. 添加 Scaler Config

创建：

```text
soma_retargeter/configs/my_robot/soma_to_my_robot_scaler_config.json
```

基础模板：

```json
{
    "robot_type": "my_robot",
    "human_root_name": "Hips",
    "human_height_assumption": 1.8,
    "joint_scales": {
        "Hips": 1.0,
        "Chest": 0.8,
        "LeftLeg": 0.8,
        "RightLeg": 0.8,
        "LeftShin": 0.8,
        "RightShin": 0.8,
        "LeftFoot": 0.8,
        "RightFoot": 0.8,
        "LeftArm": 0.8,
        "RightArm": 0.8,
        "LeftForeArm": 0.8,
        "RightForeArm": 0.8,
        "LeftHand": 0.8,
        "RightHand": 0.8
    },
    "joint_parents": {
        "Hips": "",
        "Chest": "Hips",
        "LeftLeg": "Hips",
        "RightLeg": "Hips",
        "LeftShin": "LeftLeg",
        "RightShin": "RightLeg",
        "LeftFoot": "LeftShin",
        "RightFoot": "RightShin",
        "LeftArm": "Chest",
        "RightArm": "Chest",
        "LeftForeArm": "LeftArm",
        "RightForeArm": "RightArm",
        "LeftHand": "LeftForeArm",
        "RightHand": "RightForeArm"
    },
    "joint_offsets": {
        "Hips": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "Chest": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "LeftFoot": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "RightFoot": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    }
}
```

Scaler config 决定：

- SOMA 关节缩放比例。
- 映射关节之间的父子关系。
- 每个关节到机器人目标点的平移/旋转 offset。

## 6. 生成 Rest-Aligned Scaler Offset

用已有工具从 SOMA rest pose 和机器人 rest pose 生成 offset：

```bash
python app/optimize_scaler_config.py \
  --config soma_retargeter/configs/my_robot/soma_to_my_robot_scaler_config.json \
  --bvh soma_retargeter/configs/soma/soma_zero_frame0.bvh \
  --retargeter_config soma_retargeter/configs/my_robot/soma_to_my_robot_retargeter_config.json \
  --rest_align_offsets \
  --facing_direction Maya \
  --output soma_retargeter/configs/my_robot/soma_to_my_robot_scaler_config_rest_aligned.json
```

然后把 retargeter config 指向生成后的文件：

```json
"human_robot_scaler_config": "my_robot/soma_to_my_robot_scaler_config_rest_aligned.json"
```

生成后一定要在 viewer 里检查第一帧。常见手动修正：

- 如果脚在地面下方，抬高 scaler 里的脚部 offset z。
- 如果初始躯干歪，检查 `Hips` 和 `Chest` offset。
- 如果手离机器人腕部很远，调整 hand offset，或者把 `ik_map` 映射到更合适的 wrist/hand link。

## 7. 添加 Feet Stabilizer Config

如果是双足机器人，建议添加：

```text
soma_retargeter/configs/my_robot/my_robot_feet_stabilizer_config.json
```

示例：

```json
{
    "robot_type": "my_robot",
    "robot_model": {
        "format": "urdf",
        "path": "my_robot/my_robot.urdf",
        "floating": true,
        "collapse_fixed_joints": false,
        "hide_visuals": true,
        "parse_visuals_as_colliders": false
    },
    "ik_iterations": 20,
    "joint_limit_weight": 10.0,

    "effectors": {
        "pelvis": [30.0, 6.0],
        "left_hip_roll_link": [1.5, 0.1],
        "left_knee_link": [1.0, 0.3],
        "left_ankle_link": [10.0, 0.35],
        "right_hip_roll_link": [1.5, 0.1],
        "right_knee_link": [1.0, 0.3],
        "right_ankle_link": [10.0, 0.35]
    },

    "ik_root": 0,
    "ik_limbs": {
        "left_leg": {
            "effectors": [1, 2, 3],
            "hint_reference": 3,
            "hint_offset": [0.25, 0.0, 0.25]
        },
        "right_leg": {
            "effectors": [4, 5, 6],
            "hint_reference": 6,
            "hint_offset": [0.25, 0.0, 0.25]
        }
    }
}
```

然后在 retargeter config 里打开：

```json
"feet_stabilizer_config": "my_robot/my_robot_feet_stabilizer_config.json",
"enable_post_processing": true
```

注意：feet stabilizer 和主 IK 最好使用同一个 `robot_model`，保证 `joint_q` 顺序一致。模型不一致时，可能会导致关节数据解释错位。

## 8. 添加 CSV 输出支持

如果机器人可以使用通用 URDF movable-joint 顺序，可以编辑：

```text
soma_retargeter/assets/csv.py
```

添加：

```python
if target == "my_robot":
    urdf_path = io_utils.get_config_file("my_robot", "my_robot.urdf")
    return URDFRobotCSVConfig(
        name="my_robot",
        joint_names=get_movable_joint_names_from_urdf(str(urdf_path)),
    )
```

如果机器人需要自定义输出顺序、单位、字段名，就参考 `UnitreeG129DOF_CSVConfig` 写一个新的 `RobotCSVConfig`。

## 9. 添加 Viewer/Batch App Config

创建：

```text
assets/default_my_robot_bvh_to_csv_converter_config.json
```

示例：

```json
{
    "import_folder": "assets/motions/bvh",
    "export_folder": "assets/motions/my_robot-export",
    "batch_size": 100,
    "retargeter": "Newton",
    "retarget_source": "soma",
    "retarget_target": "my_robot",
    "retarget_source_facing_direction": "Maya"
}
```

打开 OpenGL viewer：

```bash
python ./app/bvh_to_csv_converter.py \
  --config ./assets/default_my_robot_bvh_to_csv_converter_config.json \
  --viewer gl
```

批量/headless 运行：

```bash
python ./app/bvh_to_csv_converter.py \
  --config ./assets/default_my_robot_bvh_to_csv_converter_config.json \
  --viewer null
```

## 10. 验证清单

调动作质量之前，先做这些基础检查。

检查 JSON：

```bash
python -m json.tool soma_retargeter/configs/my_robot/soma_to_my_robot_retargeter_config.json >/dev/null
python -m json.tool soma_retargeter/configs/my_robot/soma_to_my_robot_scaler_config_rest_aligned.json >/dev/null
```

检查机器人能否构建：

```bash
python - <<'PY'
from soma_retargeter.robotics.robot_loader import create_robot_builder

builder = create_robot_builder("my_robot")
model = builder.finalize()
print("body_count", builder.body_count)
print("joint_coord_count", model.joint_coord_count)
PY
```

跑一次 headless smoke test：

```bash
python ./app/bvh_to_csv_converter.py \
  --config ./assets/default_my_robot_bvh_to_csv_converter_config.json \
  --viewer null
```

再打开 viewer 肉眼检查：

```bash
python ./app/bvh_to_csv_converter.py \
  --config ./assets/default_my_robot_bvh_to_csv_converter_config.json \
  --viewer gl
```

## 常见问题

### Link 名字对不上

现象：构建 target mapping 时出现 `ValueError`。

处理：检查机器人 body 名，然后更新 retargeter config 里的 `ik_map`。

### 机器人初始脚在地面下面

现象：第 0 帧脚穿地。

处理：抬高 scaler config 里对应脚的 offset z。

### 动作抖动

常见原因：

- IK target 对低自由度机器人约束过强。
- 脚或手的 rotation weight 太高。
- 机器人接近关节限位。

处理方式：

- 降低 foot/hand 的 `r_weight`。
- 对明显不可达的脚目标，降低 `t_weight`。
- 适度增大 `output_smoothing_window`，例如 `5`、`7`、`9`。
- 如果经常撞限位，需要加机器人专属 reachability guard。

### 深蹲或跳跃时机器人后仰/翻身

原因：机器人无法在不超过膝、髋、踝或躯干限位的情况下追上源动作脚目标。比如机器人膝盖可弯角度比人体或另一个机器人小，就可能靠 pelvis/root 倾斜来补偿。

处理方式：

- 在不可达姿态中放松脚位置目标。
- 优先保住 pelvis/chest 姿态，而不是强行追脚。
- 降低 leg/foot 的 scale。
- 添加 reachability-aware target clamping。

### Feet stabilizer 反而让效果变差

原因可能是 stabilizer 的模型和主 IK 模型不一致，或者机器人的腿链不符合当前 two-bone 假设。

处理方式：

- retargeter 和 feet stabilizer 使用同一个 `robot_model`。
- 检查 `effectors` 顺序和 `ik_limbs` 索引。
- 对 ankle DOF 少的机器人，降低 ankle rotation 权重。

## 推荐接入顺序

1. 只加载机器人模型。
2. 只重定向 `Hips`、`Chest`、`LeftFoot`、`RightFoot`。
3. 生成 rest-aligned scaler offset。
4. 检查第 0 帧姿态和脚底高度。
5. 添加腿部映射。
6. 添加手臂和手部映射。
7. 调 IK objective 权重。
8. 添加 feet stabilizer。
9. 添加 CSV 输出支持。
10. 批量跑一小组 BVH，检查失败案例。

