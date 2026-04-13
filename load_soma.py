#!/usr/bin/env python3
"""Load SOMA BVH motion file and render with SOMA (following official reference)."""

import numpy as np
import torch
from bvh import Bvh, SkeletonBvh, parse_bvh_motion
from soma import SOMALayer, get_assets_dir
import trimesh

SOMA_BVH_PATH = 'sample_soma/jump_and_land_heavy_001__A001.bvh'
SOMA_PARAMS_PATH = 'assets/soma_shapes/soma_base_fit_mhr_params.npz'

def main():
    print("=" * 60)
    print("Loading SOMA BVH Motion File")
    print("=" * 60)

    # Load SOMA shape parameters
    shape_params = np.load(SOMA_PARAMS_PATH)
    identity_params = torch.from_numpy(shape_params["identity_params"])
    scale_params = torch.from_numpy(shape_params["scale_params"])

    print(f"Identity params shape: {identity_params.shape}")
    print(f"Scale params shape: {scale_params.shape}")

    # Parse BVH motion (120 fps)
    local_rot_mats, root_trans = parse_bvh_motion(SOMA_BVH_PATH)
    print(f"\nMotion loaded:")
    print(f"  local_rot_mats shape: {local_rot_mats.shape}")  # [num_frames, 77, 3, 3]
    print(f"  root_trans shape: {root_trans.shape}")  # [num_frames, 3]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # SOMA model setup (using SOMALayer as shown in official example)
    soma = SOMALayer(
        data_root=get_assets_dir(),
        identity_model_type="mhr",
        device=device,
    )

    # Move to device
    identity_params = identity_params.to(device)
    scale_params = scale_params.to(device)
    local_rot_mats = local_rot_mats.to(identity_params)
    root_trans = root_trans.to(identity_params)

    # Process first frame
    first_rot = local_rot_mats[0:1]  # (1, 77, 3, 3)
    first_trans = root_trans[0:1]    # (1, 3)

    print(f"\nFirst frame:")
    print(f"  Rotation shape: {first_rot.shape}")
    print(f"  Translation: {first_trans[0].cpu().numpy()}")

    # Run SOMA forward (using absolute_pose=True for BVH motions from BONES-SEED)
    with torch.no_grad():
        output = soma(
            first_rot,
            identity_params,
            scale_params=scale_params,
            transl=first_trans,
            pose2rot=False,       # Rotations already in matrix form
            absolute_pose=True,   # BVH poses are absolute, not wrt SOMA canonical T-pose
        )

    vertices = output["vertices"][0].cpu().numpy()  # (V, 3)
    joints = output["joints"][0].cpu().numpy()       # (J, 3)

    print(f"\nRendered mesh:")
    print(f"  Vertices shape: {vertices.shape}")
    print(f"  Joints shape: {joints.shape}")

    # Visualize with trimesh
    print("\nVisualizing first frame...")
    try:
        # Create point cloud from vertices
        cloud = trimesh.PointCloud(vertices)

        # Build skeleton lines from BVH hierarchy
        skeleton = SkeletonBvh()
        skeleton.load_from_bvh(SOMA_BVH_PATH)

        skel_edges = []
        for bone in skeleton.bones:
            if bone.parent is not None:
                skel_edges.append([bone.id, bone.parent.id])

        # Draw skeleton lines
        if skel_edges:
            lines = []
            for i, j in skel_edges:
                if i < len(joints) and j < len(joints):
                    lines.append([joints[i], joints[j]])
            lines = np.array(lines)
            if len(lines) > 0:
                skeleton_lines = trimesh.load_path(lines.reshape(-1, 2, 3))
                scene = trimesh.Scene([cloud, skeleton_lines])
            else:
                scene = trimesh.Scene([cloud])
        else:
            scene = trimesh.Scene([cloud])

        scene.show()
        print("Visualization launched!")

    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
        print("\nVertices preview (first 10):")
        print(vertices[:10])

    print("\n" + "=" * 60)
    print("SOMA BVH loaded and first frame rendered successfully!")
    print("=" * 60)

    return output, vertices, joints

if __name__ == '__main__':
    main()
