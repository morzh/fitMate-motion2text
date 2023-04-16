#!/usr/bin/env python
# coding: utf-8

import time
import pandas as pd
from multiprocessing import Pool
from human_body_prior.body_model.body_model import BodyModel

from os.path import join as pjoin

from common.skeleton import Skeleton
from common.quaternion import *
from paramUtil import *
import numpy as np
import torch
from tqdm import tqdm
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trans_matrix = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0]])

male_bm_path = './body_models/smplh/male/model.npz'
male_dmpl_path = './body_models/dmpls/male/model.npz'

female_bm_path = './body_models/smplh/female/model.npz'
female_dmpl_path = './body_models/dmpls/female/model.npz'

num_betas = 10  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls,
                    dmpl_fname=male_dmpl_path
                    ).to(device)
female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls,
                      dmpl_fname=female_dmpl_path).to(device)

data_dir = './joints'  # standard action points with augmentation
index_path = './index.csv'  # file from HumanML3D project

save_dir = './HumanML3D_40fps/'
# final action points. Only skelete.
save_dir1 = pjoin(save_dir, 'new_joints')
# final action data in 2d array of information calculated from the points
save_dir2 = pjoin(save_dir, 'new_joint_vecs')
goal_fps = 40


def step1(group_path, n_process=5):
    """ Recalculate points from original dataset to ??? """
    torch.multiprocessing.set_start_method('spawn')
    with Pool(n_process) as p:
        p.map(recalc_points, group_path)


def step1v2():
    """ Recalculate points from original dataset to ??? """
    group_path = found_file_paths()
    for paths in group_path:
        cur_count = 0
        dataset_name = paths[0].split('/')[2]
        pbar = tqdm(paths)
        pbar.set_description('Processing: %s' % dataset_name)
        fps = 0
        for path in pbar:
            save_path = path.replace('./amass_data', './pose_data')
            save_path = save_path[:-3] + 'npy'
            if not os.path.isfile(save_path):
                fps = amass_to_pose(path, save_path)

        cur_count += len(paths)
        print('Processed / All (fps %d): %d/%d' % (fps, cur_count, len(paths)))
        time.sleep(0.5)


def step2():
    """ Make motion augmentation (swap by vertical) """
    os.makedirs(data_dir, exist_ok=True)
    index_file = pd.read_csv(index_path)
    total_amount = index_file.shape[0]
    upsample = goal_fps / 20

    for i in tqdm(range(total_amount)):
        source_path = index_file.loc[i]['source_path']
        new_name = index_file.loc[i]['new_name']
        if not os.path.isfile(source_path) or os.path.isfile(pjoin(data_dir, new_name)):
            continue
        data = np.load(source_path)
        start_frame = int(index_file.loc[i]['start_frame'] *upsample)
        end_frame = int(index_file.loc[i]['end_frame']*upsample)
        if 'humanact12' not in source_path:
            if 'Eyes_Japan_Dataset' in source_path:
                data = data[3 * goal_fps:]
            if 'MPI_HDM05' in source_path:
                data = data[3 * goal_fps:]
            if 'TotalCapture' in source_path:
                data = data[1 * goal_fps:]
            if 'MPI_Limits' in source_path:
                data = data[1 * goal_fps:]
            if 'Transitions_mocap' in source_path:
                data = data[int(0.5 * goal_fps):]
            data = data[start_frame:end_frame]
            data[..., 0] *= -1

        data_m = swap_left_right(data)
        #     save_path = pjoin(data_dir, )
        np.save(pjoin(data_dir, new_name), data)
        np.save(pjoin(data_dir, 'M' + new_name), data_m)


def step3():
    """
    Recalculate the motion points again - delete hands.
    """

    joints_num = 22  # count of joints after run

    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    source_list = os.listdir(data_dir)
    frame_num = 0
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        try:
            data, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
            rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
            np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())
            np.save(pjoin(save_dir2, source_file), data)
            frame_num += data.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 20 / 60))


def found_file_paths():
    paths = []
    folders = []
    dataset_names = []
    for root, dirs, files in os.walk('./amass_data'):
        folders.append(root)
        for name in files:
            dataset_name = root.split('/')[2]
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)
            paths.append(os.path.join(root, name))

    save_folders = [folder.replace('./amass_data', './pose_data') for folder in folders]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)
    group_path = [[path for path in paths if name in path] for name in dataset_names]
    return group_path


def amass_to_pose(src_path, save_path):
    global male_bm, female_bm
    bdata = np.load(src_path, allow_pickle=True)
    fps = 0
    try:
        fps = bdata['mocap_framerate']
    except:
        return fps

    if bdata['gender'] == 'male':
        bm = male_bm
    else:
        bm = female_bm
    down_sample = int(fps / goal_fps)
    if down_sample < 1:
        down_sample = 1

    with torch.no_grad():
        root_orient = torch.Tensor(bdata['poses'][::down_sample, :3]).to(
            device)  # controls the global root orientation
        pose_body = torch.Tensor(bdata['poses'][::down_sample, 3:66]).to(
            device)  # controls the body
        pose_hand = torch.Tensor(bdata['poses'][::down_sample, 66:]).to(
            device)  # controls the finger articulation
        betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(device)  # controls the body shape
        trans = torch.Tensor(bdata['trans'][::down_sample]).to(device)
        body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas.repeat(len(root_orient), 1),
                  root_orient=root_orient)
        joint_loc = body.Jtr + torch.swapaxes(trans.unsqueeze(0), 0, 1)

    pose_seq_np = joint_loc.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)

    np.save(save_path, pose_seq_np_n)
    return fps


# This will take a few hours for all datasets, here we take one dataset as an example
# To accelerate the process, you could run multiple scripts like this at one time.
def recalc_points(paths):
    cur_count = 0
    dataset_name = paths[0].split('/')[2]
    pbar = tqdm(paths)
    pbar.set_description('Processing: %s' % dataset_name)
    fps = 0
    for path in pbar:
        save_path = path.replace('./amass_data', './pose_data')
        save_path = save_path[:-3] + 'npy'
        if not os.path.isfile(save_path):
            fps = amass_to_pose(path, save_path)

    cur_count += len(paths)
    print('Processed / All (fps %d): %d/%d' % (fps, cur_count, len(paths)))
    time.sleep(0.5)


# The above code will extract poses from **AMASS** dataset, and put them under directory **"./pose_data"**
# The source data from **HumanAct12** is already included in **"./pose_data"** in this repository. You need to **unzip** it right in this folder.
# ## Segment, Mirror and Relocate Motions
def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def process_file(positions, feet_thre):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)
    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    return data, global_positions, positions, l_velocity


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def step4(data_dir, save_dir, joints_num):
    """ Calculate mean and std for using it in training. """
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4 + (joints_num - 1) * 3] = Std[4: 4 + (joints_num - 1) * 3].mean() / 1.0
    Std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = Std[4 + (joints_num - 1) * 3: 4 + (
            joints_num - 1) * 9].mean() / 1.0
    Std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = Std[4 + (
            joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3:] = Std[4 + (
            joints_num - 1) * 9 + joints_num * 3:].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std


if __name__ == '__main__':
    step1v2()
    step2()

    # all of this variables must be not here 
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    # Get offsets of target skeleton
    example_data = np.load(os.path.join(data_dir, '000021.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain  # magic variable from noting
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    step3()
    step4(data_dir, save_dir, 22)
