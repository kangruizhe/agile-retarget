import json
import argparse
import pathlib
import sys
import warp as wp
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import soma_retargeter.utils.pose_utils as pose_utils
import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.utils.newton_utils as newton_utils
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str
import newton
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline

wp.init()


def _quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_canonical(q):
    q = _quat_normalize(q)
    if q[3] < 0.0:
        q = -q
    return q


def _quat_inverse(q):
    q = _quat_normalize(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul_raw(a, b):
    ax, ay, az, aw = np.asarray(a, dtype=np.float64)
    bx, by, bz, bw = np.asarray(b, dtype=np.float64)
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _quat_mul(a, b):
    return _quat_normalize(_quat_mul_raw(_quat_normalize(a), _quat_normalize(b)))


def _quat_rotate(q, v):
    q = _quat_normalize(q)
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return _quat_mul_raw(_quat_mul_raw(q, vq), _quat_inverse(q))[:3]


def _quat_rotate_inverse(q, v):
    return _quat_rotate(_quat_inverse(q), v)


def _write_json(config, output_file):
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def generate_rest_aligned_scaler_config(
    config_file,
    bvh_file,
    retargeter_config_file,
    output_file=None,
    facing_direction="Maya",
):
    if retargeter_config_file is None:
        raise ValueError("--retargeter_config is required when using --rest_align_offsets.")

    print(f"[INFO] Loading scaler config: {config_file}")
    config = io_utils.load_json(config_file)
    print(f"[INFO] Loading retargeter config: {retargeter_config_file}")
    retargeter_config = io_utils.load_json(retargeter_config_file)

    print(f"[INFO] Loading rest BVH: {bvh_file}")
    skeleton, anim = bvh_utils.load_bvh(bvh_file)
    root_xform = SpaceConverter(get_facing_direction_type_from_str(facing_direction)).transform(wp.transform_identity())
    skeleton_instance = SkeletonInstance(skeleton, [0, 0, 0], root_xform)
    skeleton_instance.set_local_transforms(anim.get_local_transforms(0))
    global_pose = skeleton_instance.compute_global_transforms()

    scaler = HumanToRobotScaler(skeleton, retargeter_config['model_height'], config_file)
    mapped_joint_indices = scaler.mapped_joint_indices.numpy()
    mapped_joint_scales = scaler.mapped_joint_scales.numpy()
    current_effectors = scaler.compute_effectors_from_skeleton(skeleton_instance, True)

    print("[INFO] Extracting robot rest pose from Newton model...")
    pipeline = NewtonPipeline(skeleton, robot_type=config.get('robot_type', 'unitree_g1'), retarget_config=retargeter_config)
    model = pipeline.ik_model
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    body_q = state.body_q.numpy()
    body_names = [newton_utils.get_name_from_label(label) for label in pipeline.robot_builder.body_label]

    root_pose = global_pose[int(mapped_joint_indices[0])]
    root_t = np.asarray(root_pose[0:3], dtype=np.float64)
    scaled_root_t = root_t * float(mapped_joint_scales[0])
    root_joint_name = scaler.mapped_joints[0]
    ik_map = retargeter_config.get('ik_map', {})
    root_mapping = ik_map.get(root_joint_name, {})
    robot_root_body = root_mapping.get('t_body', body_names[0])
    robot_root_t = np.asarray(body_q[body_names.index(robot_root_body)][0:3], dtype=np.float64)
    target_root_t = np.asarray(current_effectors[0][0:3], dtype=np.float64)

    print(
        f"[INFO] Anchoring robot rest structure to current {root_joint_name} target "
        f"at [{target_root_t[0]:.4f}, {target_root_t[1]:.4f}, {target_root_t[2]:.4f}] m "
        f"instead of the robot model origin.")

    position_errors = []
    rotation_errors = []
    updated = []

    for i, joint_name in enumerate(scaler.mapped_joints):
        if joint_name not in ik_map:
            continue

        mapping_data = ik_map[joint_name]
        t_body = mapping_data['t_body']
        r_body = mapping_data['r_body']
        if t_body not in body_names or r_body not in body_names:
            print(f"[WARNING] Skipping {joint_name}: missing body link {t_body!r} or {r_body!r}.")
            continue

        human_pose = global_pose[int(mapped_joint_indices[i])]
        human_t = np.asarray(human_pose[0:3], dtype=np.float64)
        human_q = _quat_normalize(human_pose[3:7])

        robot_link_t = np.asarray(body_q[body_names.index(t_body)][0:3], dtype=np.float64)
        target_t = target_root_t + (robot_link_t - robot_root_t)
        target_q = _quat_normalize(body_q[body_names.index(r_body)][3:7])

        offset_q = _quat_canonical(_quat_mul(_quat_inverse(human_q), target_q))
        effector_q = _quat_mul(human_q, offset_q)

        scaled_relative_t = (human_t - root_t) * float(mapped_joint_scales[i])
        offset_t = _quat_rotate_inverse(effector_q, target_t - scaled_relative_t - scaled_root_t)

        predicted_t = scaled_relative_t + scaled_root_t + _quat_rotate(effector_q, offset_t)
        predicted_q = _quat_mul(human_q, offset_q)

        position_errors.append(float(np.linalg.norm(predicted_t - target_t)))
        rotation_errors.append(float(min(
            np.linalg.norm(predicted_q - target_q),
            np.linalg.norm(predicted_q + target_q),
        )))

        config['joint_offsets'][joint_name] = [
            [float(offset_t[0]), float(offset_t[1]), float(offset_t[2])],
            [float(offset_q[0]), float(offset_q[1]), float(offset_q[2]), float(offset_q[3])],
        ]
        updated.append((joint_name, float(np.linalg.norm(offset_t))))

    if output_file is None:
        output_file = config_file.replace('.json', '_rest_aligned.json')

    _write_json(config, output_file)

    print(f"[INFO] Updated {len(updated)} IK-mapped joint offsets:")
    for joint_name, offset_len in updated:
        print(f"[INFO]\t{joint_name:12s} |offset_t| = {offset_len:.4f} m")

    if position_errors:
        print(f"[INFO] Rest position check: mean={np.mean(position_errors):.8f} m, max={np.max(position_errors):.8f} m")
        print(f"[INFO] Rest rotation check: mean={np.mean(rotation_errors):.8f}, max={np.max(rotation_errors):.8f}")
    print(f"[INFO] Saved rest-aligned scaler config to: {output_file}")
    return output_file

@wp.kernel
def compute_bone_length_loss(
    predicted_effectors: wp.array(dtype=wp.transform),
    target_effectors: wp.array(dtype=wp.transform),
    mapped_ancestors: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    if mask[tid] < 0.5:
        return
        
    parent_id = mapped_ancestors[tid]
    
    if parent_id >= 0:
        pred_t = wp.transform_get_translation(predicted_effectors[tid])
        pred_parent_t = wp.transform_get_translation(predicted_effectors[parent_id])
        pred_dist = wp.length(pred_t - pred_parent_t)
        
        targ_t = wp.transform_get_translation(target_effectors[tid])
        targ_parent_t = wp.transform_get_translation(target_effectors[parent_id])
        targ_dist = wp.length(targ_t - targ_parent_t)
        
        diff = pred_dist - targ_dist
        err = diff * diff
        wp.atomic_add(loss, 0, err)

@wp.kernel
def compute_position_loss(
    predicted_effectors: wp.array(dtype=wp.transform),
    target_effectors: wp.array(dtype=wp.transform),
    pred_root: wp.array(dtype=wp.vec3),
    targ_root: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    if mask[tid] < 0.5:
        return
    if tid == 0:
        pred_t = wp.transform_get_translation(predicted_effectors[tid])
        targ_t = wp.transform_get_translation(target_effectors[tid])
    else:
        pred_t = wp.transform_get_translation(predicted_effectors[tid]) - pred_root[0]
        targ_t = wp.transform_get_translation(target_effectors[tid]) - targ_root[0]
    diff = pred_t - targ_t
    wp.atomic_add(loss, 0, wp.dot(diff, diff))

def optimize_scaler(
    config_file,
    bvh_file,
    retargeter_config_file=None,
    iterations=100,
    learning_rate=0.01,
    facing_direction="Maya",
    output_file=None,
    offset_lr_scale=0.02,
    phase2_patience=30,
):
    print(f"[INFO] Loading scaler config: {config_file}")
    config = io_utils.load_json(config_file)
    retargeter_config = None
    if retargeter_config_file:
        retargeter_config = io_utils.load_json(retargeter_config_file)
    model_height = (
        retargeter_config.get('model_height')
        if retargeter_config is not None and retargeter_config.get('model_height') is not None
        else config.get('model_height', 1.70)
    )
    print(f"[INFO] Using model_height={model_height:.3f} for scaler optimization.")
    
    # Load BVH
    print(f"[INFO] Loading target BVH: {bvh_file}")
    skeleton, anim = bvh_utils.load_bvh(bvh_file)
    root_xform = SpaceConverter(get_facing_direction_type_from_str(facing_direction)).transform(wp.transform_identity())
    skeleton_instance = SkeletonInstance(skeleton, [0, 0, 0], root_xform)
    skeleton_instance.set_local_transforms(anim.get_local_transforms(0)) # Use first frame
    
    # Initialize scaler to get mapping info
    scaler = HumanToRobotScaler(skeleton, model_height, config_file)
    
    # Check mapped joints and create a gradient mask
    mask_np = np.ones(len(scaler.mapped_joints), dtype=np.float32)
    if retargeter_config is not None:
        print(f"[INFO] Using retargeter config to mask non-mapped joints: {retargeter_config_file}")
        ik_map_joints = retargeter_config.get('ik_map', {}).keys()
        for i, joint_name in enumerate(scaler.mapped_joints):
            if joint_name not in ik_map_joints:
                mask_np[i] = 0.0
                print(f"[INFO] Ignored optimization for unmapped joint: {joint_name}")
    
    update_mask = wp.array(mask_np, dtype=wp.float32)
    
    # 1. Prepare Traiable parameters (Scales and Offsets)
    # Convert joint scales to wp.array with requires_grad=True
    mapped_joint_scales = wp.array(
        scaler.mapped_joint_scales.numpy(), 
        dtype=wp.float32, 
        requires_grad=True
    )
    
    # We also optimize joint offsets if needed. Here we optimize position and rotation of offsets
    mapped_joint_offsets = wp.array(
        scaler.mapped_joint_offsets.numpy(), 
        dtype=wp.transform, 
        requires_grad=True
    )
    mapped_joint_rotation_modes = scaler.mapped_joint_rotation_modes

    # 2. Get global pose of skeleton
    num_joints = skeleton_instance.num_joints
    wp_global_pose = wp.array([wp.transform_identity()] * num_joints, dtype=wp.transform, requires_grad=True)

    # Note: `pose_utils.wp_compute_global_pose` inside `HumanToRobotScaler` is wrapped in a kernel. Let's make our own.
    @wp.kernel
    def _compute_global_pose_kernel(
        in_num_joints: wp.int32,
        in_root_tx: wp.transform,
        in_parent_indices: wp.array(dtype=wp.int32),
        in_local_pose: wp.array(dtype=wp.transform),
        out_result: wp.array(dtype=wp.transform)
    ):
        pose_utils.wp_compute_global_pose(in_num_joints, in_root_tx, in_parent_indices, in_local_pose, out_result)

    wp.launch(
        _compute_global_pose_kernel,
        dim=1,
        inputs=[
            num_joints, skeleton_instance.xform, 
            wp.array(skeleton_instance.parent_indices, dtype=wp.int32), 
            wp.array(skeleton_instance.local_transforms, dtype=wp.transform)
        ],
        outputs=[wp_global_pose]
    )
    
    # 3. Define target effectors from the Robot's Rest Pose
    target_effectors_np = scaler.compute_effectors_from_skeleton(skeleton_instance, scale_animation=True)
    
    if retargeter_config is not None:
        print("[INFO] Extracting actual robot rest pose to use as optimization targets...")
        pipeline = NewtonPipeline(skeleton, robot_type=config.get('robot_type', 'unitree_g1'), retarget_config=retargeter_config)
        model = pipeline.ik_model
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()
        
        body_names = [newton_utils.get_name_from_label(label) for label in pipeline.robot_builder.body_label]
        
        for i, joint_name in enumerate(scaler.mapped_joints):
            if joint_name in ik_map_joints:
                t_body = retargeter_config['ik_map'][joint_name]['t_body']
                r_body = retargeter_config['ik_map'][joint_name]['r_body']
                if t_body in body_names and r_body in body_names:
                    t_idx = body_names.index(t_body)
                    r_idx = body_names.index(r_body)
                    
                    link_pos = body_q[t_idx][0:3]
                    link_rot = body_q[r_idx][3:7]
                    
                    # Store exact rest pose coordinates of the robot effector into the target
                    target_effectors_np[i][0:3] = link_pos
                    target_effectors_np[i][3:7] = link_rot
                else:
                    print(f"[WARNING] Could not find body link '{t_body}' or '{r_body}' in robot model for {joint_name}.")
    else:
        # Fallback dummy shift
        print("[WARNING] No retargeter config provided. Using dummy 5cm perturbation as target.")
        for i in range(len(target_effectors_np)):
            target_effectors_np[i][1] += 0.05
            
    target_effectors = wp.array(target_effectors_np, dtype=wp.transform)
    
    def get_mapped_parent(idx, parents, mask_arr):
        p = parents[idx]
        while p >= 0 and mask_arr[p] < 0.5:
            p = parents[p]
        return p

    mapped_ancestors_np = np.zeros(len(scaler.mapped_joints), dtype=np.int32)
    for i in range(len(scaler.mapped_joints)):
        mapped_ancestors_np[i] = get_mapped_parent(i, scaler.mapped_joint_parents, mask_np)
        
    wp_mapped_ancestors = wp.array(mapped_ancestors_np, dtype=wp.int32)
    
    # Shared root arrays for position loss
    wp_targ_root = wp.array([target_effectors_np[0][0:3]], dtype=wp.vec3)

    # 4. Optimization – Phase 1: Optimize scales via bone-length loss
    print(f"[INFO] Phase 1 – Scale optimization for {iterations} iterations...")
    history_loss = []
    
    @wp.kernel
    def step_kernel_float(param: wp.array(dtype=wp.float32), grad: wp.array(dtype=wp.float32), mask: wp.array(dtype=wp.float32), lr: float):
        param[wp.tid()] = param[wp.tid()] - lr * grad[wp.tid()] * mask[wp.tid()]

    @wp.kernel
    def step_kernel_vec3_offsets(
        offsets: wp.array(dtype=wp.transform),
        grad_offsets: wp.array(dtype=wp.transform),
        mask: wp.array(dtype=wp.float32),
        lr: float
    ):
        tid = wp.tid()
        if mask[tid] < 0.5:
            return
        g_t = wp.transform_get_translation(grad_offsets[tid])
        cur_t = wp.transform_get_translation(offsets[tid])
        cur_q = wp.transform_get_rotation(offsets[tid])
        new_t = cur_t - lr * g_t
        offsets[tid] = wp.transform(new_t, cur_q)

    for i in range(iterations):
        tape = wp.Tape()
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        wp_effectors = wp.zeros(len(scaler.mapped_joint_indices), dtype=wp.transform, requires_grad=True)
        with tape:
            wp.launch(kernel_compute_scaled_effectors, dim=1,
                inputs=[len(scaler.mapped_joint_indices), wp_global_pose, scaler.mapped_joint_indices,
                        mapped_joint_scales, mapped_joint_offsets, mapped_joint_rotation_modes, True],
                outputs=[wp_effectors])
            wp.launch(compute_bone_length_loss, dim=len(scaler.mapped_joint_indices),
                inputs=[wp_effectors, target_effectors, wp_mapped_ancestors, update_mask, loss])
        tape.backward(loss)
        l_val = loss.numpy()[0]
        history_loss.append(l_val)
        if i % 10 == 0 or i == iterations - 1:
            print(f"[Phase1] Iter {i:4d} | BoneLen Loss: {l_val:10.6f}")
        wp.launch(step_kernel_float, dim=len(scaler.mapped_joint_indices),
            inputs=[mapped_joint_scales, tape.gradients[mapped_joint_scales], update_mask, learning_rate])
        tape.zero()

    # Phase 2: Optimize offset translations via per-joint position loss
    print(f"[INFO] Phase 2 – Offset translation optimization for {iterations} iterations...")
    lr_offset = learning_rate * offset_lr_scale   # smaller LR for offsets
    print(f"[INFO] Phase 2 offset learning rate: {lr_offset:.6f}")
    best_phase2_loss = None
    best_phase2_offsets = mapped_joint_offsets.numpy().copy()
    phase2_no_improve = 0
    for i in range(iterations):
        tape = wp.Tape()
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        wp_effectors = wp.zeros(len(scaler.mapped_joint_indices), dtype=wp.transform, requires_grad=True)
        wp_pred_root = wp.zeros(1, dtype=wp.vec3, requires_grad=False)
        with tape:
            wp.launch(kernel_compute_scaled_effectors, dim=1,
                inputs=[len(scaler.mapped_joint_indices), wp_global_pose, scaler.mapped_joint_indices,
                        mapped_joint_scales, mapped_joint_offsets, mapped_joint_rotation_modes, True],
                outputs=[wp_effectors])
            # Capture root
            @wp.kernel
            def _get_root(eff: wp.array(dtype=wp.transform), out: wp.array(dtype=wp.vec3)):
                out[0] = wp.transform_get_translation(eff[0])
            wp.launch(_get_root, dim=1, inputs=[wp_effectors], outputs=[wp_pred_root])
            wp.launch(compute_position_loss, dim=len(scaler.mapped_joint_indices),
                inputs=[wp_effectors, target_effectors, wp_pred_root, wp_targ_root, update_mask, loss])
        tape.backward(loss)
        l_val = loss.numpy()[0]
        if best_phase2_loss is None or l_val < best_phase2_loss:
            best_phase2_loss = l_val
            best_phase2_offsets = mapped_joint_offsets.numpy().copy()
            phase2_no_improve = 0
        else:
            phase2_no_improve += 1
        if i % 10 == 0 or i == iterations - 1:
            print(f"[Phase2] Iter {i:4d} | Position Loss: {l_val:10.6f}")
        if phase2_patience > 0 and phase2_no_improve >= phase2_patience:
            print(
                f"[INFO] Phase 2 early stop at iter {i}; "
                f"best Position Loss: {best_phase2_loss:10.6f}"
            )
            break
        wp.launch(step_kernel_vec3_offsets, dim=len(scaler.mapped_joint_indices),
            inputs=[mapped_joint_offsets, tape.gradients[mapped_joint_offsets], update_mask, lr_offset])
        tape.zero()

    if best_phase2_loss is not None:
        mapped_joint_offsets = wp.array(best_phase2_offsets, dtype=wp.transform, requires_grad=True)
        print(f"[INFO] Restored best Phase2 offsets with Position Loss: {best_phase2_loss:10.6f}")
        wp_effectors = wp.zeros(len(scaler.mapped_joint_indices), dtype=wp.transform, requires_grad=True)
        wp.launch(kernel_compute_scaled_effectors, dim=1,
            inputs=[len(scaler.mapped_joint_indices), wp_global_pose, scaler.mapped_joint_indices,
                    mapped_joint_scales, mapped_joint_offsets, mapped_joint_rotation_modes, True],
            outputs=[wp_effectors])

    print("[INFO] Optimization complete.")
    
    # Save optimized parameters back to config
    # HumanToRobotScaler applies a ratio on load, we must divide it out to save properly.
    human_height_assump = config.get('human_height_assumption', 1.80)
    ratio = model_height / human_height_assump
    
    optimized_scales = mapped_joint_scales.numpy()
    for idx, name in enumerate(scaler.mapped_joints):
        config['joint_scales'][name] = float(optimized_scales[idx]) / ratio
    
    # Save optimized offset translations (divide ratio out of translation too)
    optimized_offsets = mapped_joint_offsets.numpy()  # shape: (N, 7) [px,py,pz, qx,qy,qz,qw]
    for idx, name in enumerate(scaler.mapped_joints):
        orig_name = name
        if orig_name in ('LeftToeBase', 'RightToeBase'):
            orig_name = orig_name.replace('ToeBase', 'Toe')
        if orig_name in config.get('joint_offsets', {}):
            t = optimized_offsets[idx][0:3]
            q = list(config['joint_offsets'][orig_name][1])  # keep rotation unchanged
            config['joint_offsets'][orig_name] = [[float(t[0]), float(t[1]), float(t[2])], q]
        
    if output_file is None:
        output_file = config_file.replace('.json', '_optimized.json')
    _write_json(config, output_file)
        
    print(f"[INFO] Saved optimized scaler config to: {output_file}")
    
    # Plot Optimization Results
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10))
        
        # 3D Alignment
        ax2 = fig.add_subplot(111, projection='3d')
        final_effectors_np = wp_effectors.numpy()
        
        # Subtract root position so both skeletons share the same origin
        targ_root = target_effectors_np[0][0:3].copy()
        pred_root = final_effectors_np[0][0:3].copy()
        
        targ_x, targ_y, targ_z = [], [], []
        pred_x, pred_y, pred_z = [], [], []
        
        for i, name in enumerate(scaler.mapped_joints):
            if mask_np[i] > 0.5: # Only plot optimized/mapped joints
                targ_p = target_effectors_np[i][0:3] - targ_root
                pred_p = final_effectors_np[i][0:3] - pred_root
                
                targ_x.append(targ_p[0])
                targ_y.append(targ_p[1])
                targ_z.append(targ_p[2])
                
                pred_x.append(pred_p[0])
                pred_y.append(pred_p[1])
                pred_z.append(pred_p[2])
                
        # Draw the hierarchical structural lines separately
        for i in range(len(scaler.mapped_joints)):
            if mask_np[i] > 0.5:
                parent_idx = mapped_ancestors_np[i]
                if parent_idx >= 0:
                    # Robot lines
                    t_p = target_effectors_np[parent_idx][0:3] - targ_root
                    t_c = target_effectors_np[i][0:3] - targ_root
                    ax2.plot([t_p[0], t_c[0]], [t_p[1], t_c[1]], [t_p[2], t_c[2]], 'g-', alpha=0.6)
                    
                    # Human lines
                    p_p = final_effectors_np[parent_idx][0:3] - pred_root
                    p_c = final_effectors_np[i][0:3] - pred_root
                    ax2.plot([p_p[0], p_c[0]], [p_p[1], p_c[1]], [p_p[2], p_c[2]], 'r-', alpha=0.6)
                
        ax2.scatter(targ_x, targ_y, targ_z, c='green', marker='s', s=45, label='Robot Target (MJCF)')
        ax2.scatter(pred_x, pred_y, pred_z, c='red', marker='o', s=45, label='Soma Scaled (Optimized T-Pose)')
        
        ax2.set_title('Structural Alignment (Robot Rest Pose vs SOMA T-Pose)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        # Uniform axis scaling
        all_x = targ_x + pred_x
        all_y = targ_y + pred_y
        all_z = targ_z + pred_z
        mid_x = (max(all_x) + min(all_x)) / 2
        mid_y = (max(all_y) + min(all_y)) / 2
        mid_z = (max(all_z) + min(all_z)) / 2
        half_range = max(max(all_x)-min(all_x), max(all_y)-min(all_y), max(all_z)-min(all_z)) / 2
        ax2.set_xlim(mid_x - half_range, mid_x + half_range)
        ax2.set_ylim(mid_y - half_range, mid_y + half_range)
        ax2.set_zlim(mid_z - half_range, mid_z + half_range)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = "optimization_alignment.png"
        plt.savefig(plot_path)
        print(f"[INFO] Optimization alignment plot saved to: {plot_path}")
    except ImportError:
        print("[WARNING] matplotlib is not installed. Skipping visualization.")


@wp.kernel
def kernel_compute_scaled_effectors(
    in_num_mapped_joints    : wp.int32,
    in_global_pose          : wp.array(dtype=wp.transform),
    in_mapped_joint_indices : wp.array(dtype=wp.int32),
    in_mapped_joint_scales  : wp.array(dtype=wp.float32),
    in_mapped_joint_offsets : wp.array(dtype=wp.transform),
    in_mapped_joint_rotation_modes : wp.array(dtype=wp.int32),
    in_scale_animation      : wp.bool,
    out_result              : wp.array(dtype=wp.transform)
):
    HumanToRobotScaler.wp_compute_scaled_effectors(
        in_num_mapped_joints, in_global_pose, in_mapped_joint_indices,
        in_mapped_joint_scales, in_mapped_joint_offsets, in_mapped_joint_rotation_modes,
        in_scale_animation, out_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize joint scales in scaler config using Warp.")
    parser.add_argument("--config", type=str, required=True, help="Path to the scaler config JSON file.")
    parser.add_argument("--bvh", type=str, required=True, help="Path to a BVH file used as reference human motion.")
    parser.add_argument("--retargeter_config", type=str, default=None, help="Path to retargeter config JSON file to mask joints not mapped in IK.")
    parser.add_argument("--iters", type=int, default=100, help="Number of gradient descent iterations.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--facing_direction", type=str, default="Maya", choices=["Maya", "Mujoco"], help="Source facing direction used by the runtime converter.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path. Defaults to *_optimized.json or *_rest_aligned.json.")
    parser.add_argument("--rest_align_offsets", action="store_true", help="Generate offset rotations/translations from SOMA rest pose to robot rest pose, without gradient optimization.")
    parser.add_argument("--offset_lr_scale", type=float, default=0.02, help="Phase 2 offset LR multiplier relative to --lr.")
    parser.add_argument("--phase2_patience", type=int, default=30, help="Stop Phase 2 after this many non-improving iterations. Use 0 to disable.")
    
    args = parser.parse_args()
    if args.rest_align_offsets:
        generate_rest_aligned_scaler_config(
            args.config,
            args.bvh,
            args.retargeter_config,
            args.output,
            args.facing_direction)
    else:
        optimize_scaler(
            args.config,
            args.bvh,
            args.retargeter_config,
            args.iters,
            args.lr,
            args.facing_direction,
            args.output,
            args.offset_lr_scale,
            args.phase2_patience)
