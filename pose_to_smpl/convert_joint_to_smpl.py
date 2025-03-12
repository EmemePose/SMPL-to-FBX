
import os
import glob
import random
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import pickle
from scipy.spatial.transform import Rotation as R
import cv2


import smplx
from mpl_toolkits.mplot3d import Axes3D

# edges=model.per_skeleton_joint_edges['smpl_24'].numpy()
"""
this edge was different from the smpl_bones
in smpl, the spine3 (9) is the parent of neck (12) and the arms (13, 14)
but in this edges, the spine3 (9) is the parent of neck (12) only, and head (12) is the parent of the arms (13, 14)

update:
NO, based on the FBX file, the neck (12) should be the parent of the arms (13, 14)
"""


SMPL_24_LABELS = np.array([b'pelv', b'lhip', b'rhip', b'spi1', b'lkne', b'rkne', b'spi2',
	   b'lank', b'rank', b'spi3', b'ltoe', b'rtoe', b'neck', b'lcla',
	   b'rcla', b'head', b'lsho', b'rsho', b'lelb', b'relb', b'lwri',
	   b'rwri', b'lhan', b'rhan'])



smpl_bones = [
		[ 0,  1],
		[ 0,  2],
		[ 0,  3],
		[ 1,  4],
		[ 2,  5],
		[ 3,  6],
		[ 4,  7],
		[ 5,  8],
		[ 6,  9],
		[ 7, 10],
		[ 8, 11],
		[ 9, 12],  ## [9, 13], [9,14]
		[12, 13],
		[12, 14],
		[12, 15],
		[13, 16],
		[14, 17],
		[16, 18], 
		[17, 19],
		[18, 20],
		[19, 21],
		[20, 22],
		[21, 23],
		]

SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 9, 10, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21])

smpl_bone_length = {
	(0, 1): [ 0.066, -0.087, -0.002],  # Pelvis → Left Hip
	(0, 2): [-0.066, -0.087, -0.002],  # Pelvis → Right Hip
	(0, 3): [ 0.000,  0.105,  0.000],  # Pelvis → Spine1
	(1, 4): [ 0.003, -0.398,  0.005],  # Left Hip → Left Knee
	(2, 5): [-0.003, -0.398,  0.005],  # Right Hip → Right Knee
	(3, 6): [ 0.000,  0.128,  0.000],  # Spine1 → Spine2
	(4, 7): [ 0.006, -0.376,  0.008],  # Left Knee → Left Ankle
	(5, 8): [-0.006, -0.376,  0.008],  # Right Knee → Right Ankle
	(6, 9): [ 0.000,  0.132,  0.000],  # Spine2 → Spine3
	(7, 10): [ 0.000, -0.100,  0.000],  # Left Ankle → Left Toe
	(8, 11): [ 0.000, -0.100,  0.000],  # Right Ankle → Right Toe

	(9, 12): [ 0.000,  0.182,  0.000],  # Spine3 → Neck

	# (9, 13): [ 0.080,  0.155, -0.008],  # Spine3 → Left Clavicle
	# (9, 14): [-0.080,  0.155, -0.008],  # Spine3 → Right Clavicle

	(12, 13): [ 0.080,  0.155, -0.008],  # Neck → Left Clavicle
	(12, 14): [-0.080,  0.155, -0.008],  # Neck → Right Clavicle

	(12, 15): [ 0.000,  0.073,  0.000],  # Neck → Head
	(13, 16): [ 0.128,  0.015, -0.012],  # Left Clavicle → Left Shoulder
	(14, 17): [-0.128,  0.015, -0.012],  # Right Clavicle → Right Shoulder
	(16, 18): [ 0.281, -0.015, -0.001],  # Left Shoulder → Left Elbow
	(17, 19): [-0.281, -0.015, -0.001],  # Right Shoulder → Right Elbow
	(18, 20): [ 0.235,  0.000, -0.001],  # Left Elbow → Left Wrist
	(19, 21): [-0.235,  0.000, -0.001],  # Right Elbow → Right Wrist
	(20, 22): [ 0.072,  0.000,  0.000],  # Left Wrist → Left Hand
	(21, 23): [-0.072,  0.000,  0.000],  # Right Wrist → Right Hand
	}

smpl_bone_length = {k: np.array(v) * 2000 for k, v in smpl_bone_length.items()}
# smpl_bones=[
# 		(0,1), (1,4), (4,7), (7,10), # R leg
# 	   (0,2), (2,5), (5,8), (8,11), # L leg
# 	   (0,3), (3,6), (6,9), # Spine
# 	   (9,12), (12,15), # Head
# 	   (9,13), (13,16), (16,18), (18,20), (20,22), # R arm
# 	   (9,14), (14,17), (17,19), (19,21), (21,23)] # L arm

# SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])





def rotation_matrix(x, y, z):
	'''
	Get the rotation matrix in opencv coordinate system
	x, y, z: roll, pitch, yaw, (radians)
	'''
	Rx = np.array([[1,0,0],
				[0, np.cos(x), -np.sin(x)],
				[0, np.sin(x), np.cos(x)]])

	Ry = np.array([[ np.cos(y), 0, np.sin(y)],
				[ 0,         1,         0],
				[-np.sin(y), 0, np.cos(y)]])

	Rz = np.array([[np.cos(z), -np.sin(z), 0],
				[np.sin(z),  np.cos(z), 0],
				[0,0,1]])
	return Rz@Ry@Rx



R_x = rotation_matrix(np.pi, 0, 0)




def visualize_3d_pose_with_labels(
	poses3d, smpl_bones, output_path="output.mp4", fps=30, frame_size=(10, 10), show_labels=True
):

	poses3d_centered = poses3d - poses3d[:, :1, :]  # Subtract the root joint position

	# Apply a 90° rotation around the X-axis
	# Swap Y and Z to account for Matplotlib's different convention
	poses3d_centered[..., 1], poses3d_centered[..., 2] = poses3d_centered[..., 2], -poses3d_centered[..., 1]

	writer = FFMpegWriter(fps=fps)
	fig = plt.figure(figsize=frame_size)
	ax = fig.add_subplot(111, projection="3d")

	with writer.saving(fig, output_path, dpi=100):
		for frame_idx, pose3d in enumerate(poses3d_centered):
			ax.cla()

			x_range = np.max(np.abs(poses3d_centered[:, :, 0]))
			y_range = np.max(np.abs(poses3d_centered[:, :, 1]))
			z_range = np.max(np.abs(poses3d_centered[:, :, 2]))
			max_range = max(x_range, y_range, z_range)
			ax.set_xlim3d(-max_range, max_range)
			ax.set_ylim3d(-max_range, max_range)
			ax.set_zlim3d(-max_range, max_range)

			ax.set_xlabel("X-axis")
			ax.set_ylabel("Y-axis")
			ax.set_zlabel("Z-axis")

			for i_start, i_end in smpl_bones:
				ax.plot(
					[pose3d[i_start, 0], pose3d[i_end, 0]],
					[pose3d[i_start, 1], pose3d[i_end, 1]],
					[pose3d[i_start, 2], pose3d[i_end, 2]],
					marker="o",
					markersize=4,
					linewidth=2,
				)

			ax.scatter(
				pose3d[:, 0],
				pose3d[:, 1],
				pose3d[:, 2],
				c="red",
				s=30,
				label="Keypoints",
			)

			if show_labels:
				for i, (x, y, z) in enumerate(pose3d):
					ax.text(x, y, z, SMPL_24_LABELS[i], color='black', fontsize=10, horizontalalignment='center')
			ax.set_title(f"Frame {frame_idx + 1}")
			writer.grab_frame()

	plt.close(fig)
	print(f"Video saved to {output_path}")

# def convert_pose3d_to_smpl(pose3d):
# 	"""
# 	Convert pose3d (N, 24, 3) to SMPL-compatible (N, 72) pose parameters.
# 	+ pelvis-relative transformations.
# 	+ Converts local joint orientations to axis-angle rotations.
# 	"""
# 	num_frames = pose3d.shape[0]
# 	smpl_poses = np.zeros((num_frames, 72))
	
# 	# pose3d = pose3d - pose3d[:, :1, :]  # Subtract pelvis joint

# 	pose3d[..., 1], pose3d[..., 2] = pose3d[..., 2], -pose3d[..., 1]

# 	# pose3d = pose3d @ R_x.T

# 	for i in  tqdm(range(num_frames)):
# 		joints = pose3d[i]  # Shape (24, 3)

# 		# pelvis = joints[0].copy()
# 		# rel_joints = joints - pelvis  # put pelvis at (0,0,0)

# 		rotations = []
# 		for j in range(0, 24):
# 			parent_idx = SMPL_PARENTS[j]

# 			if parent_idx == -1:
# 				rotation = np.eye(3)
# 			else:
# 				child_vec = joints[j] - joints[parent_idx]

# 				parent_vec = np.array([0, 0, 1])


# 				# nonzero vector
# 				if np.linalg.norm(child_vec) < 1e-6:
# 					rotation = np.eye(3)
# 				else:
# 					child_vec /= np.linalg.norm(child_vec)  # Normalize
# 					print("child_vec", child_vec)
# 					print("parent_vec", parent_vec)
# 					rotation = R.align_vectors([child_vec], [parent_vec])[0].as_matrix()

# 			rotations.append(rotation)

# 		#  to axis-angle
# 		smpl_poses[i] = np.concatenate([R.from_matrix(rot).as_rotvec() for rot in rotations])

# 	return smpl_poses

def convert_pose3d_to_smpl(pose3d):
	"""
	Convert 3D joint positions (N, 24, 3) to SMPL-compatible (N, 72) pose parameters.
	"""
	num_frames = pose3d.shape[0]
	smpl_poses = np.zeros((num_frames, 72))

	# Make root joint (pelvis) the origin
	# pose3d = pose3d - pose3d[:, :1, :]

	# Apply coordinate transformation
	pose3d = pose3d @ R_x.T  

	# Initialize a placeholder for previous frame joint vectors
	prev_frame_joints = None

	for i in tqdm(range(num_frames)):
		joints = pose3d[i]  # Shape (24, 3)
		rotations = []

		for j in range(24):
			parent_idx = SMPL_PARENTS[j]

			if parent_idx == -1:
				rotation = np.eye(3)  # Root pelvis stays identity
			else:
				child_vec = joints[j] - joints[parent_idx]  # Current frame joint vector

				# Use previous frame vector if available; else, assume identity
				if prev_frame_joints is not None:
					parent_vec = prev_frame_joints[j] - prev_frame_joints[parent_idx]
				else:
					parent_vec = np.array([0.0, 1.0, 0.0])  # Safe default


				# parent_vec = np.array([1, 0, 0])
				

				# Avoid zero-length vectors
				if np.linalg.norm(child_vec) < 1e-6 or np.linalg.norm(parent_vec) < 1e-6:
					rotation = np.eye(3)
				else:
					# Normalize vectors
					child_vec /= np.linalg.norm(child_vec)
					parent_vec /= np.linalg.norm(parent_vec)

					# Compute rotation matrix
					rotation = R.align_vectors([child_vec], [parent_vec])[0].as_matrix()

			rotations.append(rotation)

		# Save current frame joint positions for the next iteration
		prev_frame_joints = joints.copy()

		# Convert rotation matrices to axis-angle and store them
		smpl_poses[i] = np.concatenate([R.from_matrix(rot).as_rotvec() for rot in rotations])
	print(smpl_poses.shape)
	for i in range(num_frames):
		print(smpl_poses[i])
	return smpl_poses





def plot_smpl(smpl_poses, smpl_trans, output_video="smpl_animation.mp4", fps=30):
	""" Visualize and animate SMPL motion in 3D """

	num_frames = smpl_poses.shape[0]
	
	# Set up the Matplotlib figure
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection="3d")
	
	writer = FFMpegWriter(fps=fps)
	
	# Start recording the animation
	with writer.saving(fig, output_video, dpi=100):
		for frame_idx in range(num_frames):
			ax.cla()  # Clear the plot
			
			# Get the current pose and translation
			pose = smpl_poses[frame_idx].reshape(24, 3)  # Convert to (24, 3) axis-angle
			trans = smpl_trans[frame_idx]  # Translation of the root
			
			# Convert pose from axis-angle to rotation matrices
			rotation_matrices = R.from_rotvec(pose).as_matrix()  # Shape: (24, 3, 3)

			# Compute joint positions from rotations
			joints = np.zeros((24, 3))  # Store joint positions
			joints[0] = trans  # Root joint

			# Compute joint locations using forward kinematics
			for parent, child in smpl_bones:
				# joints[child] = joints[parent] + 500 * rotation_matrices[child][:, 1]  # Move along Y-axis

				bone_vector = smpl_bone_length[(parent, child)]  # Get predefined bone vector
				rotated_bone_vector = rotation_matrices[parent] @ bone_vector  # Apply rotation
				joints[child] = joints[parent] + rotated_bone_vector  # Move child joint

			

			# Plot joints
			ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', s=30)

			# Plot bones
			for parent, child in smpl_bones:
				ax.plot(
					[joints[parent, 0], joints[child, 0]],
					[joints[parent, 1], joints[child, 1]],
					[joints[parent, 2], joints[child, 2]],
					color='blue', linewidth=2
				)

			ax.scatter(
				joints[:, 0],
				joints[:, 1],
				joints[:, 2],
				c="red",
				s=30,
				label="Keypoints",
			)

			for i, (x, y, z) in enumerate(joints):
				ax.text(x, y, z, SMPL_24_LABELS[i], color='black', fontsize=10, horizontalalignment='center')


			# Set axis limits dynamically
			max_range = np.max(np.abs(joints))
			ax.set_xlim3d(-max_range, max_range)
			ax.set_ylim3d(-max_range, max_range)
			ax.set_zlim3d(-max_range, max_range)

			ax.set_xlabel("X-axis")
			ax.set_ylabel("Y-axis")
			ax.set_zlabel("Z-axis")
			ax.set_title(f"Frame {frame_idx + 1}")

			writer.grab_frame()  # Save the frame

	plt.close(fig)
	print(f"Video saved: {output_video}")

if __name__ == '__main__':
	root= '/home/jqin/Ememe/pose_to_smpl'


	#### read a correct smpl file ####
	with open('/home/jqin/Ememe/SMPL-to-FBX/data/pkl/gBR_sBM_cAll_d04_mBR0_ch01.pkl', 'rb') as f:
		smpl_pkl = pickle.load(f)

	smpl_poses = smpl_pkl['smpl_poses']
	smpl_trans = smpl_pkl['smpl_trans']
	num_frames = smpl_poses.shape[0]
	for i in range(num_frames):
		# print(smpl_poses[i])
		print('trans :', smpl_trans[i])
	
	smpl_trans = smpl_trans - smpl_trans[:, [0]]

	plot_smpl(smpl_poses, smpl_trans, output_video=os.path.join( root, 'gBR_sBM_cAll_d04_mBR0_ch01.mp4' ), fps=30)
	exit(0)

	############
	

	original_pkl_path = os.path.join( root, 'testfbx_conversion_pose_3d.pkl')
	mp4_path = os.path.join( root, 'baseline_test_motiontrans.mp4' )
	# output_pkl_path = os.path.join( root, 'testfbx_conversion_smpl_.pkl' ) # the one on Notion

	output_pkl_path = '/home/jqin/Ememe/SMPL-to-FBX/data/pkl/conversion_by_jiawei.pkl'

	# load original pkl (n_frames, 24, 3)
	with open(original_pkl_path, 'rb') as f:
		pose3d = pickle.load(f)

	print(pose3d.shape)
	
	# visualize_3d_pose_with_labels(pose3d, smpl_bones, output_path=mp4_path, fps=30, frame_size=(10, 10), show_labels=True)


	# Convert to SMPL (joint rotation) format, like joint order in https://github.com/softcat477/SMPL-to-FBX/tree/main
	smpl_poses = convert_pose3d_to_smpl(pose3d)
	smpl_trans = pose3d[:, 0, :]


	plot_smpl(smpl_poses, smpl_trans, output_video=os.path.join( root, 'smpl_animation.mp4' ), fps=30)

	# Save as a new .pkl file for further FBX convertion
	output_data = {
		"smpl_poses": smpl_poses,  # (N, 72)
		"smpl_trans": smpl_trans,  # (N, 3)
		"smpl_scaling": np.array([1.0])  # Default scaling factor
	}

	with open(output_pkl_path, "wb") as f:
		pickle.dump(output_data, f)
	print("SMPL file saved to ", output_pkl_path)




# def convert_to_root_relative(motion_data, root_joint_idx=0):
# 	"""
# 	Convert motion data to root-relative by subtracting the root joint's position.
# 	"""
# 	root_positions = motion_data[:, root_joint_idx, :]

# 	root_relative_motion = motion_data - root_positions[:, np.newaxis, :]

# 	return root_relative_motion


# def reproject_3d_pose_simple(pose_3d, image_width, image_height, focal_length=1500):
# 	"""
# 	Reprojects a 3D pose based on image dimensions without using bounding boxes.
# 	"""
# 	cx, cy = image_width / 2, image_height / 2

# 	pose_3d_normalized = pose_3d.copy()
# 	pose_3d_normalized[..., 0] = (pose_3d[..., 0] - cx) / focal_length  # X-axis
# 	pose_3d_normalized[..., 1] = (pose_3d[..., 1] - cy) / focal_length  # Y-axis
# 	pose_3d_normalized[..., 2] = pose_3d[..., 2] / focal_length  # Depth normalization

# 	return pose_3d_normalized


# def load_pkl_as_array(file_path):
# 	"""Loads a .pkl file and returns its contents as a NumPy array.
# 	"""
# 	try:
# 		with open(file_path, 'rb') as f:
# 			data = pickle.load(f)
# 			# Assuming the .pkl file contains a single NumPy array
# 			if isinstance(data, np.ndarray):
# 				return data
# 			else:
# 				print("Warning: The .pkl file does not contain a NumPy array.")
# 				return data
# 	except (FileNotFoundError, pickle.UnpicklingError) as e:
# 		print(f"Error loading .pkl file: {e}")
# 		return None
  

# def pred_tranform(preds):
# 	poses3d_list=[]
# 	for p in preds:
# 		poses3d_list.append(p['poses3d'])#.numpy()
# 	detections_list=[]
# 	for p in preds:
# 		detections_list.append(p['boxes'])#.numpy()
# 	return poses3d_list, detections_list


# def convert_boxes_format(original_boxes):
#     """
#     Converts a list of (1, 5) shaped arrays into a (num_frames, 4) array.
#     """
#     # Extract only the first four values (x_min, y_min, x_max, y_max) from each array
#     boxes_array = np.array([box[0, :4] for box in original_boxes], dtype=np.float32)
#     return boxes_array
