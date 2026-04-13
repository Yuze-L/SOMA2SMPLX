#!/usr/bin/env python3
"""Convert SOMA motion to SMPL-X format using Barycentric Interpolation."""

import numpy as np
import torch
import trimesh
from bvh import parse_bvh_motion
from soma import SOMALayer, get_assets_dir
from soma.geometry.barycentric_interp import BarycentricInterpolator
from smplfitter.pt import BodyModel, BodyFitter

# Paths
SOMA_BVH_PATH = 'sample_soma/jump_and_land_heavy_001__A001.bvh'
SOMA_PARAMS_PATH = 'assets/soma_shapes/soma_base_fit_mhr_params.npz'
SOMA_ASSETS_DIR = get_assets_dir()

def main():
    print("=" * 60)
    print("SOMA to SMPL-X Conversion using Barycentric Interpolation")
    print("=" * 60)

    # Load correspondence meshes for SOMA <-> SMPL-X conversion
    print("\nLoading SOMA <-> SMPL-X correspondence meshes...")
    mesh_smplx = trimesh.load(f"{SOMA_ASSETS_DIR}/SMPLX/base_body.obj", maintain_order=True, process=False)
    mesh_soma = trimesh.load(f"{SOMA_ASSETS_DIR}/SMPLX/SOMA_wrap.obj", maintain_order=True, process=False)

    V_smplx = torch.from_numpy(mesh_smplx.vertices).float()
    V_soma = torch.from_numpy(mesh_soma.vertices).float()
    F_smplx = torch.from_numpy(mesh_smplx.faces).long()
    F_soma = torch.from_numpy(mesh_soma.faces).long()

    print(f"SMPL-X template: {V_smplx.shape[0]} vertices, {F_smplx.shape[0]} faces")
    print(f"SOMA wrap mesh: {V_soma.shape[0]} vertices, {F_soma.shape[0]} faces")

    # Create SOMA -> SMPL-X interpolator
    print("\nCreating SOMA -> SMPL-X interpolator...")
    soma_to_smplx = BarycentricInterpolator(V_soma, F_soma, V_smplx)
    print("Interpolator created successfully")

    # Load SOMA shape parameters
    print("\nLoading SOMA parameters...")
    shape_params = np.load(SOMA_PARAMS_PATH)
    identity_params = torch.from_numpy(shape_params["identity_params"])
    scale_params = torch.from_numpy(shape_params["scale_params"])

    # Parse BVH motion
    local_rot_mats, root_trans = parse_bvh_motion(SOMA_BVH_PATH)
    print(f"\nMotion: {local_rot_mats.shape[0]} frames at 120 FPS")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create SOMA model
    soma = SOMALayer(
        data_root=SOMA_ASSETS_DIR,
        identity_model_type="mhr",
        device=device,
    )

    # Move data to device
    identity_params = identity_params.to(device)
    scale_params = scale_params.to(device)
    local_rot_mats = local_rot_mats.to(identity_params)
    root_trans = root_trans.to(identity_params)

    # Process first frame
    first_rot = local_rot_mats[0:1]  # (1, 77, 3, 3)
    first_trans = root_trans[0:1]    # (1, 3)

    # Run SOMA forward to get mesh
    print("\nRendering SOMA mesh...")
    with torch.no_grad():
        soma_output = soma(
            first_rot,
            identity_params,
            scale_params=scale_params,
            transl=first_trans,
            pose2rot=False,
            absolute_pose=True,
        )

    soma_vertices = soma_output["vertices"][0]  # (18056, 3)
    soma_joints = soma_output["joints"][0]    # (77, 3)

    print(f"SOMA Output:")
    print(f"  Vertices: {soma_vertices.shape}")
    print(f"  Joints: {soma_joints.shape}")

    # Convert SOMA vertices to SMPL-X vertices using Barycentric Interpolation
    print("\nConverting SOMA vertices to SMPL-X format...")
    soma_vertices_np = soma_vertices.cpu()  # Ensure on CPU for interpolator
    smplx_vertices = soma_to_smplx(soma_vertices_np)  # Should be (10475, 3)
    print(f"Converted SMPL-X vertices shape: {smplx_vertices.shape}")

    # Now we have SMPL-X format vertices - use smplfitter to get SMPL-X parameters
    print("\nSetting up SMPL-X fitter...")

    # Create SMPL-X body model
    body_model = BodyModel(
        model_name='smplx',
        gender='neutral',
        model_root='assets/body_models/smplx',
        num_betas=10,
        device=device,
    )

    # Create body fitter
    fitter = BodyFitter(body_model)

    # Prepare target vertices for fitting (in SMPL-X topology)
    target_vertices = smplx_vertices.unsqueeze(0)  # (1, 10475, 3)
    print(f"Target vertices for fitting: {target_vertices.shape}")

    # Fit SMPL-X to the converted vertices
    print("\nFitting SMPL-X to converted vertices...")
    fit_result = fitter.fit(
        target_vertices=target_vertices,
        num_iter=3,
        beta_regularizer=1.0,
        beta_regularizer2=0.1,
        share_beta=True,
        scale_fit=True,
    )

    # Extract fitted parameters
    fitted_betas = fit_result['shape_betas']  # (1, 10)
    fitted_trans = fit_result['trans']         # (1, 3)
    fitted_pose = fit_result['pose_rotvecs']    # (1, 55*3)
    fitted_orient = fit_result['orientations']  # Global orientations

    print(f"\nFitted SMPL-X parameters:")
    print(f"  Shape betas: {fitted_betas.shape}")
    print(f"  Beta values: {fitted_betas[0].cpu().numpy()}")
    print(f"  Translation: {fitted_trans[0].cpu().numpy()}")
    print(f"  Pose rotvecs shape: {fitted_pose.shape}")

    # Generate SMPL-X mesh with fitted parameters
    print("\nGenerating SMPL-X mesh with fitted parameters...")
    with torch.no_grad():
        smpl_output = body_model(
            pose_rotvecs=fitted_pose,
            shape_betas=fitted_betas,
            trans=fitted_trans,
        )

    smpl_vertices = smpl_output['vertices'][0].cpu().numpy()
    smpl_joints = smpl_output['joints'][0].cpu().numpy()

    print(f"SMPL-X Output:")
    print(f"  Vertices: {smpl_vertices.shape}")
    print(f"  Joints: {smpl_joints.shape}")

    # Visualize comparison
    print("\nVisualizing comparison...")
    try:
        # SOMA mesh (blue) - converted to SMPL-X topology
        soma_smplx_cloud = trimesh.PointCloud(smplx_vertices.numpy(), colors=[100, 200, 255])

        # Fitted SMPL mesh (orange)
        smpl_cloud = trimesh.PointCloud(smpl_vertices, colors=[255, 150, 100])

        scene = trimesh.Scene([soma_smplx_cloud, smpl_cloud])
        scene.show()

        print("Visualization launched!")
        print("  Blue: SOMA mesh (converted to SMPL-X topology)")
        print("  Orange: Fitted SMPL-X mesh")

    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("SOMA to SMPL-X conversion complete!")
    print("=" * 60)

    return {
        'soma_vertices': soma_vertices.cpu().numpy(),
        'smplx_vertices': smplx_vertices.numpy(),
        'smpl_vertices': smpl_vertices,
        'fitted_betas': fitted_betas[0].cpu().numpy(),
        'fitted_pose': fitted_pose[0].cpu().numpy(),
        'fitted_trans': fitted_trans[0].cpu().numpy(),
    }

if __name__ == '__main__':
    main()