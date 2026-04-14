# Basic usage
python soma2smpl_converter.py \
    --input sample_soma/jump_and_land_heavy_001__A001.bvh \
    --shape-params assets/soma_shapes/soma_base_fit_mhr_params.npz \
    --output converted_motion.npz

# With visualization
python soma2smpl_converter.py \
    --input sample_soma/jump_and_land_heavy_001__A001.bvh \
    --shape-params assets/soma_shapes/soma_base_fit_mhr_params.npz \
    --output converted_motion.npz \
    --visualize

# Process specific frames
python soma2smpl_converter.py \
    --input sample_soma/jump_and_land_heavy_001__A001.bvh \
    --shape-params assets/soma_shapes/soma_base_fit_mhr_params.npz \
    --output converted_motion.npz \
    --num-frames 100 --start-frame 50
Output format (npz file contains):

pose: (num_frames, 165) - SMPL-X pose as rotation vectors
trans: (num_frames, 3) - translation
betas: (num_frames, 10) - shape parameters
vertices: (num_frames, 10475, 3) - mesh vertices
joints: (num_frames, 55, 3) - joint positions
frame_time: scalar - 1/120 seconds
num_frames: scalar - number of frames