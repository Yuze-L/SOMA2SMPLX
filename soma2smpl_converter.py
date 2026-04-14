#!/usr/bin/env python3
"""Convert SOMA motion to SMPL-X format using Barycentric Interpolation.

Usage:
    python soma2smpl_converter.py \\
        --input sample_soma/jump_and_land_heavy_001__A001.bvh \\
        --shape-params assets/soma_shapes/soma_base_fit_mhr_params.npz \\
        --output converted_motion.npz
"""

import argparse
import numpy as np
import pickle
import torch
import trimesh
from bvh import parse_bvh_motion
from soma import SOMALayer, get_assets_dir
from soma.geometry.barycentric_interp import BarycentricInterpolator
from smplfitter.pt import BodyModel, BodyFitter

# Default paths
DEFAULT_SOMA_ASSETS_DIR = get_assets_dir()


def setup_interpolators(soma_assets_dir):
    """Load and setup SOMA <-> SMPL-X correspondence meshes."""
    mesh_smplx = trimesh.load(f"{soma_assets_dir}/SMPLX/base_body.obj", maintain_order=True, process=False)
    mesh_soma = trimesh.load(f"{soma_assets_dir}/SMPLX/SOMA_wrap.obj", maintain_order=True, process=False)

    V_smplx = torch.from_numpy(mesh_smplx.vertices).float()
    V_soma = torch.from_numpy(mesh_soma.vertices).float()
    F_smplx = torch.from_numpy(mesh_smplx.faces).long()
    F_soma = torch.from_numpy(mesh_soma.faces).long()

    print(f"SMPL-X template: {V_smplx.shape[0]} vertices, {F_smplx.shape[0]} faces")
    print(f"SOMA wrap mesh: {V_soma.shape[0]} vertices, {F_soma.shape[0]} faces")

    # Create SOMA -> SMPL-X interpolator
    soma_to_smplx = BarycentricInterpolator(V_soma, F_soma, V_smplx)
    print("Interpolator created successfully")

    return soma_to_smplx


def convert_soma_to_smplx(soma_vertices, interpolator):
    """Convert SOMA vertices to SMPL-X format using Barycentric Interpolation."""
    smplx_vertices = interpolator(soma_vertices)
    return smplx_vertices


def fit_smplx_to_vertices(smplx_vertices, fitter, device):
    """Fit SMPL-X model to target vertices and return pose parameters."""
    target_vertices = smplx_vertices.unsqueeze(0).to(device)  # (1, 10475, 3)

    # Fit SMPL-X to the converted vertices
    fit_result = fitter.fit(
        target_vertices=target_vertices,
        num_iter=3,
        beta_regularizer=1.0,
        beta_regularizer2=0.1,
        share_beta=True,
        scale_fit=True,
    )

    return fit_result


def save_smplx_motion(output_path, motion_data, format='npz'):
    """Save SMPL-X motion data to file.

    Args:
        output_path: Path to save the motion file
        motion_data: Dictionary containing motion data
        format: 'npz' or 'pkl'
    """
    if format == 'npz':
        np.savez(output_path, **motion_data)
        print(f"Saved SMPL-X motion to {output_path}")
    elif format == 'pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(motion_data, f)
        print(f"Saved SMPL-X motion to {output_path}")
    else:
        raise ValueError(f"Unknown format: {format}. Use 'npz' or 'pkl'.")


def main():
    parser = argparse.ArgumentParser(description='Convert SOMA motion to SMPL-X format')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input SOMA BVH motion file')
    parser.add_argument('--shape-params', '-s', type=str, required=True,
                        help='Path to SOMA shape parameters (.npz file)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output SMPL-X motion file (.npz or .pkl)')
    parser.add_argument('--soma-assets', type=str, default=DEFAULT_SOMA_ASSETS_DIR,
                        help='Path to SOMA assets directory')
    parser.add_argument('--smplx-assets', type=str, default='assets/body_models/smplx',
                        help='Path to SMPL-X assets directory')
    parser.add_argument('--num-frames', '-n', type=int, default=None,
                        help='Number of frames to convert (default: all)')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Starting frame index (default: 0)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize the first frame after conversion')
    parser.add_argument('--save-individual-frames', action='store_true',
                        help='Save per-frame vertices and joints')

    args = parser.parse_args()

    print("=" * 60)
    print("SOMA to SMPL-X Conversion")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Shape params: {args.shape_params}")
    print(f"Output: {args.output}")
    print(f"SOMA assets: {args.soma_assets}")
    print(f"SMPLX assets: {args.smplx_assets}")

    # Setup interpolators
    print("\nLoading SOMA <-> SMPL-X correspondence meshes...")
    interpolator = setup_interpolators(args.soma_assets)

    # Load SOMA shape parameters
    print("\nLoading SOMA parameters...")
    shape_params = np.load(args.shape_params)
    identity_params = torch.from_numpy(shape_params["identity_params"])
    scale_params = torch.from_numpy(shape_params["scale_params"])

    # Parse BVH motion
    print("\nParsing BVH motion...")
    local_rot_mats, root_trans = parse_bvh_motion(args.input)

    num_frames = len(local_rot_mats)
    if args.num_frames is not None:
        num_frames = min(args.num_frames, num_frames)
    start_frame = args.start_frame
    end_frame = start_frame + num_frames

    print(f"Motion: {num_frames} frames (frames {start_frame} to {end_frame-1}) at 120 FPS")
    print(f"Root trans shape: {root_trans.shape}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create SOMA model
    soma = SOMALayer(
        data_root=args.soma_assets,
        identity_model_type="mhr",
        device=device,
    )

    # Move data to device and ensure correct dtype
    identity_params = identity_params.to(device).float()
    scale_params = scale_params.to(device).float()
    local_rot_mats = local_rot_mats.to(device).float()
    root_trans = root_trans.to(device).float()

    # Create SMPL-X body model and fitter
    print("\nSetting up SMPL-X model...")
    body_model = BodyModel(
        model_name='smplx',
        gender='neutral',
        model_root=args.smplx_assets,
        num_betas=10,
        device=str(device),
    )
    fitter = BodyFitter(body_model)

    # Process motion frames
    print(f"\nProcessing {num_frames} frames...")
    all_smplx_poses = []
    all_smplx_trans = []
    all_smplx_betas = []
    all_smplx_vertices = []
    all_smplx_joints = []

    for frame_idx in range(start_frame, end_frame):
        if frame_idx % 100 == 0:
            print(f"  Processing frame {frame_idx}/{end_frame-1}...")

        frame_rot = local_rot_mats[frame_idx:frame_idx+1]  # (1, 77, 3, 3)
        frame_trans = root_trans[frame_idx:frame_idx+1]    # (1, 3)

        # Run SOMA forward
        with torch.no_grad():
            soma_output = soma(
                frame_rot,
                identity_params,
                scale_params=scale_params,
                transl=frame_trans,
                pose2rot=False,
                absolute_pose=True,
            )

        soma_vertices = soma_output["vertices"][0]  # (18056, 3)

        # Convert SOMA vertices to SMPL-X vertices
        smplx_vertices = convert_soma_to_smplx(soma_vertices.cpu(), interpolator)

        # Fit SMPL-X to vertices
        fit_result = fit_smplx_to_vertices(smplx_vertices, fitter, device)

        # Extract fitted parameters
        fitted_betas = fit_result['shape_betas'][0].cpu().numpy()   # (10,)
        fitted_trans = fit_result['trans'][0].cpu().numpy()          # (3,)
        fitted_pose = fit_result['pose_rotvecs'][0].cpu().numpy()   # (165,) - 55*3

        all_smplx_betas.append(fitted_betas)
        all_smplx_trans.append(fitted_trans)
        all_smplx_poses.append(fitted_pose)
        all_smplx_vertices.append(smplx_vertices.numpy())

        # Also get SMPL-X joints
        with torch.no_grad():
            smpl_output = body_model(
                pose_rotvecs=torch.from_numpy(fitted_pose).unsqueeze(0).to(device),
                shape_betas=torch.from_numpy(fitted_betas).unsqueeze(0).to(device),
                trans=torch.from_numpy(fitted_trans).unsqueeze(0).to(device),
            )
        smplx_joints = smpl_output['joints'][0].cpu().numpy()
        all_smplx_joints.append(smplx_joints)

    # Convert lists to arrays
    all_smplx_betas = np.array(all_smplx_betas)        # (num_frames, 10)
    all_smplx_trans = np.array(all_smplx_trans)        # (num_frames, 3)
    all_smplx_poses = np.array(all_smplx_poses)       # (num_frames, 165)
    all_smplx_vertices = np.array(all_smplx_vertices)  # (num_frames, 10475, 3)
    all_smplx_joints = np.array(all_smplx_joints)     # (num_frames, num_joints, 3)

    print(f"\nConverted motion shape:")
    print(f"  Poses: {all_smplx_poses.shape}")
    print(f"  Translations: {all_smplx_trans.shape}")
    print(f"  Betas: {all_smplx_betas.shape}")
    print(f"  Vertices: {all_smplx_vertices.shape}")
    print(f"  Joints: {all_smplx_joints.shape}")

    # Prepare output data
    motion_data = {
        'pose': all_smplx_poses,          # (num_frames, 165) - SMPL-X pose rotvecs
        'trans': all_smplx_trans,         # (num_frames, 3) - translation
        'betas': all_smplx_betas,         # (num_frames, 10) - shape parameters
        # 'vertices': all_smplx_vertices,   # (num_frames, 10475, 3) - mesh vertices
        'joints': all_smplx_joints,       # (num_frames, num_joints, 3) - joint positions
        'fps': 120,              # Frame time in seconds
        'num_frames': num_frames,
    }

    # Save to file
    output_format = args.output.split('.')[-1].lower()
    save_smplx_motion(args.output, motion_data, format=output_format)

    # Visualize first frame if requested
    if args.visualize:
        print("\nVisualizing first frame...")
        try:
            # SOMA mesh (blue) - converted to SMPL-X topology
            soma_smplx_cloud = trimesh.PointCloud(all_smplx_vertices[0], colors=[100, 200, 255])

            # Fitted SMPL mesh (orange)
            smpl_cloud = trimesh.PointCloud(all_smplx_vertices[0], colors=[255, 150, 100])

            scene = trimesh.Scene([soma_smplx_cloud, smpl_cloud])
            scene.show()
            print("Visualization launched!")
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SOMA to SMPL-X conversion complete!")
    print("=" * 60)

    return motion_data


if __name__ == '__main__':
    main()