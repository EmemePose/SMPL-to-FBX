import numpy as np
import smplx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
def plot_smpl_joints(model_path, gender='neutral'):
    """
    Visualize the SMPL joints and bones in 3D.

    Args:
        model_path (str): Path to the SMPL model (.pkl file).
        gender (str): Gender of the SMPL model ('male', 'female', 'neutral').

    Returns:
        None
    """
    # Load the SMPL model
    
    smpl_model = smplx.create(model_path, model_type='smpl', gender=gender)
    
    # Get default joint locations (zero pose)
    body_pose = np.zeros((69,))  # 23 joints * 3 axis-angle values
    global_orient = np.zeros((3,))
    betas = np.zeros((10,))  # Shape parameters
    transl = np.zeros((3,))  # Root translation

    betas = torch.tensor(np.expand_dims(betas, axis=0), dtype=torch.float32)
    body_pose = torch.tensor(np.expand_dims(body_pose, axis=0), dtype=torch.float32)
    global_orient = torch.tensor(np.expand_dims(global_orient, axis=0), dtype=torch.float32)
    transl = torch.tensor(np.expand_dims(transl, axis=0), dtype=torch.float32)
    output = smpl_model(
        betas=betas,
        body_pose=body_pose,
        global_orient= global_orient,
        transl=transl,
        return_verts=False,
        return_full_pose=False
    )

    # Extract joint positions (batch_size=1, num_joints=24, 3)
    joints = output.joints.detach().cpu().numpy().squeeze()

    # Define the SMPL kinematic chain (bones connecting joints)
    # smpl_bones = [
    #     (0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (5, 6),  # Spine
    #     (3, 7), (7, 8), (8, 9), (9, 10), (10, 11),  # Left leg
    #     (3, 12), (12, 13), (13, 14), (14, 15), (15, 16),  # Right leg
    #     (3, 17), (17, 18), (18, 19),  # Left arm
    #     (3, 20), (20, 21), (21, 22)  # Right arm
    # ]

    smpl_bones=[(0,1), (1,4), (4,7), (7,10), # R leg
       (0,2), (2,5), (5,8), (8,11), # L leg
       (0,3), (3,6), (6,9), # Spine
       (9,12), (12,15), # Head
       (9,13), (13,16), (16,18), (18,20), (20,22), # R arm
       (9,14), (14,17), (17,19), (19,21), (21,23)] # L arm
    
    # Create a 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='red', s=40, label="Joints")

    # Plot bones by connecting the joints
    for (start, end) in smpl_bones:
        x_values = [joints[start, 0], joints[end, 0]]
        y_values = [joints[start, 1], joints[end, 1]]
        z_values = [joints[start, 2], joints[end, 2]]
        ax.plot(x_values, y_values, z_values, color='blue', linewidth=2)

    # Set labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("SMPL Joints and Bones Visualization")

    # Adjust the view
    ax.view_init(elev=15, azim=120)  # Adjust for a better view
    ax.legend()
    
    # Show plot
    plt.show()



if __name__=="__main__":
        
    # Path to your SMPL model (.pkl file)
    # smpl_model_path = "/home/jqin/Ememe/pose_to_smpl/smpl_data/SMPL_python_v.1.1.0/smpl/models/SMPL_MALE.pkl"
    smpl_model_path = "/home/jqin/Ememe/SMPL-to-FBX/data/pkl/smpl_male.pkl"
    

    # Plot SMPL joints and bones
    plot_smpl_joints(smpl_model_path, gender='male')