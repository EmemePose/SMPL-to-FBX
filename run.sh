
# cd /home/jqin/Ememe/pose_to_smpl
# python convert_joint_to_smpl.py

cd /home/jqin/Ememe/SMPL-to-FBX

python Convert.py \
    --input_pkl_base data/pkl \
    --fbx_source_path data/fbx/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx \
    --output_base output

