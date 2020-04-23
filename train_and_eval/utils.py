"""
DIP: training, evaluating and running of deep inertial poser.
Copyright (C) 2018 ETH Zurich, Emre Aksan, Manuel Kaufmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import time
import os
from tensorflow.python.client import timeline
import numpy as np
import quaternion
import cv2

import torch

from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

#                    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#                    0  1  1  1  1  1  1  0  0  1  0  0  1  1  1  1  1  1  1  1  0  0  0  0
SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
SMPL_NR_JOINTS = 24
#                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14  15  16  17  18  19  20  21  22  23
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

computing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_dir_timestamp(prefix="", suffix="", connector="_"):
    """
    Creates a directory name based on timestamp.

    Args:
        prefix:
        suffix:
        connector: one connector character between prefix, timestamp and suffix.

    Returns:

    """
    return prefix+connector+str(int(time.time()))+connector+suffix


def create_tf_timeline(model_dir, run_metadata):
    """
    This is helpful for profiling slow Tensorflow code.

    Args:
        model_dir:
        run_metadata:

    Returns:

    """
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    timeline_file_path = os.path.join(model_dir,'timeline.json')
    with open(timeline_file_path, 'w') as f:
        f.write(ctf)


def split_data_dictionary(dictionary, split_indices, keys_frozen=[], verbose=1):
    """
    Splits the data dictionary of lists into smaller chunks. All (key,value) pairs must have the same number of
    elements. If there is an index error, then the corresponding (key, value) pair is copied directly to the new
    dictionaries.

    Args:
        dictionary (dict): data dictionary.
        split_indices (list): Each element contains a list of indices for one split. Multiple splits are supported.
        keys_frozen (list): list of keys that are to be copied directly. The remaining (key,value) pairs will be
            used in splitting. If not provided, all keys will be used.
        verbose (int): status messages.

    Returns:
        (tuple): a tuple containing chunks of new dictionaries.
    """
    # Find <key, value> pairs having an entry per sample.
    sample_level_keys = []
    dataset_level_keys = []

    num_samples = sum([len(l) for l in split_indices])
    for key, value in dictionary.items():
        if not(key in keys_frozen) and ((isinstance(value, np.ndarray) or isinstance(value, list)) and (len(value) == num_samples)):
            sample_level_keys.append(key)
        else:
            dataset_level_keys.append(key)
            print(str(key) + " is copied.")

    chunks = []
    for chunk_indices in split_indices:
        dict = {}

        for key in dataset_level_keys:
            dict[key] = dictionary[key]
        for key in sample_level_keys:
            dict[key] = [dictionary[key][i] for i in chunk_indices]
        chunks.append(dict)

    return tuple(chunks)


def aa_to_rot_matrix(data):
    """
    Converts the orientation data to represent angle axis as rotation matrices. `data` is expected in format
    (seq_length, n*3). Returns an array of shape (seq_length, n*9).
    """
    # reshape to have sensor values explicit
    data_c = np.array(data, copy=True)
    seq_length, n = data_c.shape[0], data_c.shape[1] // 3
    data_r = np.reshape(data_c, [seq_length, n, 3])

    qs = quaternion.from_rotation_vector(data_r)
    rot = np.reshape(quaternion.as_rotation_matrix(qs), [seq_length, n, 9])

    return np.reshape(rot, [seq_length, 9*n])


def rot_matrix_to_aa(data):
    """
    Converts the orientation data given in rotation matrices to angle axis representation. `data` is expected in format
    (seq_length, n*9). Returns an array of shape (seq_length, n*3).
    """
    seq_length, n_joints = data.shape[0], data.shape[1]//9
    data_r = np.reshape(data, [seq_length, n_joints, 3, 3])
    data_c = np.zeros([seq_length, n_joints, 3])
    for i in range(seq_length):
        for j in range(n_joints):
            data_c[i, j] = np.ravel(cv2.Rodrigues(data_r[i, j])[0])
    return np.reshape(data_c, [seq_length, n_joints*3])


def get_seq_len_histogram(sequence_length_array, num_bins=10, collapse_first_and_last_bins=None):
    """
    Creates a histogram of sequence-length.
    Args:
        sequence_length_array: numpy array of sequence length for all samples.
        num_bins:
        collapse_first_and_last_bins: selects bin edges between the provided indices by discarding from the first and
            last bins.
    Returns:
        (list): bin edges.
    """
    collapse_first_and_last_bins = collapse_first_and_last_bins or [1, -1]
    h, bins = np.histogram(sequence_length_array, bins=num_bins)
    if collapse_first_and_last_bins is not None:
        return [int(b) for b in bins[collapse_first_and_last_bins[0]:collapse_first_and_last_bins[1]]]
    else:
        return [int(b) for b in bins]


def rot_to_aa_representation(sample_list):
    """
    Args:
        sample_list (list): of motion samples with shape of (seq_len, SMPL_NR_JOINTS*9).

    Returns:

    """
    out = [np.reshape(s, [-1, SMPL_NR_JOINTS, 3, 3]) for s in sample_list]
    aa = [quaternion.as_rotation_vector(quaternion.from_rotation_matrix(s)) for s in out]
    aa = [np.reshape(s, (-1, 72)) for s in aa]

    return aa


def rad2deg(v):
    """
    Convert from radians to degrees.
    """
    return v * 180.0 / np.pi


def padded_array_to_list(data, mask):
    """
    Converts a padded numpy array to a list of un-padded numpy arrays. `data` is expected in shape
    (n, max_seq_length, ...) and `mask` in shape (n, max_seq_length). The returned value is a list of size n, each
    element being an np array of shape (dynamic_seq_length, ...).
    """
    converted = []
    seq_lengths = np.array(np.sum(mask, axis=1), dtype=np.int)
    for i in range(data.shape[0]):
        converted.append(data[i, 0:seq_lengths[i], ...])
    return converted


def smpl_reduced_to_full(smpl_reduced):
    """
    Converts an np array that uses the reduced smpl representation into the full representation by filling in
    the identity rotation for the missing joints. Can handle either rotation input (dof = 9) or quaternion input
    (dof = 4).
    :param smpl_full: An np array of shape (seq_length, n_joints_reduced*dof)
    :return: An np array of shape (seq_length, 24*dof)
    """
    dof = smpl_reduced.shape[1] // len(SMPL_MAJOR_JOINTS)
    assert dof == 9 or dof == 4
    seq_length = smpl_reduced.shape[0]
    smpl_full = np.zeros([seq_length, SMPL_NR_JOINTS * dof])
    for idx in range(SMPL_NR_JOINTS):
        if idx in SMPL_MAJOR_JOINTS:
            red_idx = SMPL_MAJOR_JOINTS.index(idx)
            smpl_full[:, idx * dof:(idx + 1) * dof] = smpl_reduced[:, red_idx * dof:(red_idx + 1) * dof]
        else:
            if dof == 9:
                identity = np.repeat(np.eye(3, 3)[np.newaxis, ...], seq_length, axis=0)
            else:
                identity = np.concatenate([np.array([[1.0, 0.0, 0.0, 0.0]])] * seq_length, axis=0)
            smpl_full[:, idx * dof:(idx + 1) * dof] = np.reshape(identity, [-1, dof])
    return smpl_full


def smpl_rot_to_global(smpl_rotations_local):
    """
    Converts local smpl rotations into global rotations by "unrolling" the kinematic chain.
    :param smpl_rotations_local: np array of rotation matrices of shape (..., N, 3, 3), or (..., 216) where N
      corresponds to the amount of joints in SMPL (currently 24)
    :return: The global rotations as an np array of the same shape as the input.
    """
    in_shape = smpl_rotations_local.shape
    do_reshape = in_shape[-1] != 3
    if do_reshape:
        assert in_shape[-1] == 216
        rots = np.reshape(smpl_rotations_local, in_shape[:-1] + (SMPL_NR_JOINTS, 3, 3))
    else:
        rots = smpl_rotations_local

    out = np.zeros_like(rots)
    dof = rots.shape[-3]
    for j in range(dof):
        if SMPL_PARENTS[j] < 0:
            out[..., j, :, :] = rots[..., j, :, :]
        else:
            parent_rot = out[..., SMPL_PARENTS[j], :, :]
            local_rot = rots[..., j, :, :]
            out[..., j, :, :] = np.matmul(parent_rot, local_rot)

    if do_reshape:
        out = np.reshape(out, in_shape)

    return out


def joint_angle_error(predicted_pose_params, target_pose_params):
    """
    Computes the distance in joint angles between predicted and target joints for every given frame. Currently,
    this function can only handle input pose parameters represented as rotation matrices.

    :param predicted_pose_params: An np array of shape `(seq_length, dof)` where `dof` is 216, i.e. a 3-by-3 rotation
      matrix for each of the 24 joints.
    :param target_pose_params: An np array of the same shape as `predicted_pose_params` representing the target poses.
    :return: An np array of shape `(seq_length, 24)` containing the joint angle error in Radians for each joint.
    """
    seq_length, dof = predicted_pose_params.shape[0], predicted_pose_params.shape[1]
    assert dof == 216, 'unexpected number of degrees of freedom'
    assert target_pose_params.shape[0] == seq_length and target_pose_params.shape[1] == dof, 'target_pose_params must match predicted_pose_params'

    # reshape to have rotation matrices explicit
    n_joints = dof // 9
    p1 = np.reshape(predicted_pose_params, [-1, n_joints, 3, 3])
    p2 = np.reshape(target_pose_params, [-1, n_joints, 3, 3])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r1 = np.reshape(p1, [-1, 3, 3])
    r2 = np.reshape(p2, [-1, 3, 3])
    r2t = np.transpose(r2, [0, 2, 1])
    r = np.matmul(r1, r2t)

    # convert `r` to angle-axis representation and extract the angle, which is our measure of difference between
    # the predicted and target orientations
    angles = []
    for i in range(r1.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))

    return np.reshape(np.array(angles), [seq_length, n_joints])

def compute_joint_locations(pose_params):
    """
    todo(lisca): docstrings!
    """
    # Conver into axis-angle format.
    pose_params = rot_matrix_to_aa(pose_params)
    
    smplh_path = "/home/lisca/models/smplh/male/model.npz"
    dmpls_path = "/home/lisca/models/dmpls/male/model.npz"
    nr_betas = 10
    nr_dmpls = 8

    # Consider the same body shape for all tests.
    # - copied from the frame 0 of amass example.
    betas = np.array([ 
        2.2140,  2.0062,  1.7169, -1.6117,  0.5180,  1.4124, -0.1580, -0.1450, 0.0671,  1.9010]).reshape((1, -1))
    betas = torch.Tensor(betas).to(computing_device)
    
    #
    smpl_model = BodyModel(
        bm_path=smplh_path, num_betas=nr_betas, path_dmpl=dmpls_path, num_dmpls=nr_dmpls).to(computing_device)

    #
    sample_joint_locations = []
    pose_params_tqdm = tqdm(pose_params)
    for smpl_pose in pose_params_tqdm:
        pose_params_tqdm.set_description("SMPL poses -> SMPL joint locations ...     ")
        # Out of 24 joints (72 elements in axis angle, cut out the root joint and the last two joints.
        pose_body = torch.Tensor(smpl_pose[3:66].reshape((1, -1))).to(computing_device)

        # todo(lisca): Carefull!!! 
        smpl_in_pose = smpl_model(pose_body=pose_body, betas=betas)

        joints = c2c(smpl_in_pose.Jtr[0])
        sample_joint_locations.append(joints)

    return np.array(sample_joint_locations)

def joint_location_error(predicted_pose_params, target_pose_params):
    assert predicted_pose_params.shape[0] == target_pose_params.shape[0], 'target_pose_params must match predicted_pose_params'
    assert predicted_pose_params.shape[1] == target_pose_params.shape[1], 'target_pose_params must match predicted_pose_params'

    seq_length, dof = predicted_pose_params.shape[0], predicted_pose_params.shape[1]
    assert dof == 216, 'unexpected number of degrees of freedom'

    # reshape to have rotation matrices explicit
    n_joints = dof // 9

    # Put the SMPL model in predicted configuration and take the location of the joints.
    # - joint_locations_predicted
    tqdm.write("\u001b[31mcomputing predicted joint locations        :\u001b[0m")
    predicted_joint_locations = compute_joint_locations(predicted_pose_params)
    
    # Put the SMPL model in target configuration and take the locations of the joints.
    # - joint_locations_target
    tqdm.write("\u001b[31mcomputing target joint locations           :\u001b[0m")
    target_joint_locations = compute_joint_locations(target_pose_params)
    
    # [(xt - xp)**2, (yt - yp)**2, (zt - zp)**2]
    diffs_squared = np.square(target_joint_locations - predicted_joint_locations)
    print("diffs_squared {}".format(np.all(diffs_squared >= 0.0)))
    
    # [(xt - xp)**2 + (yt - yp)**2 + (zt - zp)**2]
    sum_diffs_squared = np.sum(diffs_squared, axis=2)
    # sqrt([(xt - xp)**2 + (yt - yp)**2 + (zt - zp)**2])
    joints_error = np.sqrt(sum_diffs_squared)
    
    # From all 52 joints live out the errors for fingers.
    joint_locations_error = joints_error[:, :24]

#     tqdm.write(
#         "\u001b[31mjoint_location_error               : \n%s\u001b[0m" % 
#         np.array_str(joint_locations_error[0], precision=3, suppress_small=True))
    
    return joint_locations_error

def compute_metrics(prediction, target, compute_positional_error=False):
    """
    Compute the metrics on the predictions. The function can handle variable sequence lengths for each pair of
    prediction-target array. The pose parameters can either be represented as rotation matrices (dof = 9) or
    quaternions (dof = 4)
    :param prediction: a list of np arrays of size (seq_length, 24*dof)
    :param target: a list of np arrays of size (seq_length, 24*dof)
    :param compute_positional_error: if set, the euclidean pose error is calculated which can take some time.
    """
    assert len(prediction) == len(target)
    dof = prediction[0].shape[1] // SMPL_NR_JOINTS
    assert dof == 9 or dof == 4

    # because we are interested in difference per frame, flatten inputs
    pred = np.concatenate(prediction, axis=0)
    targ = np.concatenate(target, axis=0)

    if dof == 4:
        def to_rot(x):
            seq_length = x.shape[0]
            x_ = np.reshape(x, [seq_length, -1, dof])
            x_ = quaternion.as_rotation_matrix(quaternion.as_quat_array(x_))
            return np.reshape(x_, [seq_length, -1])

        # convert quaternions to rotation matrices
        pred = to_rot(pred)
        targ = to_rot(targ)

    pred_g = smpl_rot_to_global(pred)
    targ_g = smpl_rot_to_global(targ)

    angles = joint_angle_error(pred_g, targ_g)
    mm = joint_location_error(pred_g, targ_g)

#     tqdm.write(
#         "\u001b[31mjoint_location_error               : \n%s\u001b[0m" % 
#         np.array_str(mm[0], precision=3, suppress_small=True))
#     tqdm.write(
#         "\u001b[31mjoint_position_error               : \n%s\u001b[0m" % 
#         np.array_str(angles[0], precision=3, suppress_small=True))
    
    return angles, mm

class Stats(object):
    def __init__(self, tracking_joints=None, sip_evaluation_joints=None, evaluation_joints=None, logger=None):
        self.tracking_joints = tracking_joints or list(range(SMPL_NR_JOINTS))
        self.sip_evaluation_joints = sip_evaluation_joints or list(range(SMPL_NR_JOINTS))
        self.evaluation_joints = evaluation_joints or list(range(SMPL_NR_JOINTS))
        self.logger = logger

    def __enter__(self):
        self.joint_angle_diffs = None
        self.euclidean_diffs = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(exc_type, exc_value, traceback)
            return False

        sip_stats = self.get_stats(self.sip_evaluation_joints)
        self.logger.print('\n*** SIP Evaluation Error ***\n')
        self.logger.print('considered joints              : {}\n'.format(self.sip_evaluation_joints))
        self.logger.print('average joint angle error (deg): {:.4f} (+/- {:.3f})\n'.format(sip_stats[0], sip_stats[1]))
        self.logger.print('average positional error (m)   : {:.4f} (+/- {:.3f})\n'.format(sip_stats[2], sip_stats[3]))

        tracking_stats = self.get_stats(self.tracking_joints)
        self.logger.print('\n*** Tracking Error ***\n')
        self.logger.print('considered joints              : {}\n'.format(self.tracking_joints))
        self.logger.print('average joint angle error (deg): {:.4f} (+/- {:.3f})\n'.format(tracking_stats[0], tracking_stats[1]))
        self.logger.print('average positional error (m)   : {:.4f} (+/- {:.3f})\n'.format(tracking_stats[2], tracking_stats[3]))

        eval_stats = self.get_stats(self.evaluation_joints)
        self.logger.print('\n*** Remaining Evaluation Error ***\n')
        self.logger.print('considered joints              : {}\n'.format(self.evaluation_joints))
        self.logger.print('average joint angle error (deg): {:.4f} (+/- {:.3f})\n'.format(eval_stats[0], eval_stats[1]))
        self.logger.print('average positional error (m)   : {:.4f} (+/- {:.3f})\n'.format(eval_stats[2], eval_stats[3]))

        return True

    def get_stats(self, joint_idxs):
        """
        todo(lisca): docstrings!
        """

#         tqdm.write(
#             "\u001b[31mjoint_idxs                                 : %s\u001b[0m" % 
#             np.array_str(np.array(joint_idxs), precision=3, suppress_small=True))

        total_joint_angle_diff = np.mean(rad2deg(self.joint_angle_diffs[:, joint_idxs]))
        std_per_joint = np.std(rad2deg(self.joint_angle_diffs[:, joint_idxs]), axis=0)
        joint_angle_std = np.mean(std_per_joint)

        total_mm_diff = np.mean(self.euclidean_diffs[:, joint_idxs])
        std_per_joint = np.std(self.euclidean_diffs[:, joint_idxs], axis=0)
        mm_std = np.mean(std_per_joint)

        return total_joint_angle_diff, joint_angle_std, total_mm_diff, mm_std

    def add(self, ja_diffs, euc_diffs):
        """
        todo(lisca): docstrings!
        """
        if self.joint_angle_diffs is None:
            self.joint_angle_diffs = np.zeros([0, ja_diffs.shape[-1]])
        if self.euclidean_diffs is None:
            self.euclidean_diffs = np.zeros([0, euc_diffs.shape[-1]])

        self.joint_angle_diffs = np.concatenate([self.joint_angle_diffs, ja_diffs])
        self.euclidean_diffs = np.concatenate([self.euclidean_diffs, euc_diffs])

#         tqdm.write(
#             "\u001b[31mjoint_angle_diffs                          : \n%s\u001b[0m" %
#             np.array_str(len(self.joint_angle_diffs), precision=3, suppress_small=True))
#         tqdm.write(
#             "\u001b[31meuclidean_diffs                            : \n%s\u001b[0m" %
#             np.array_str(len(self.euclidean_diffs), precision=3, suppress_small=True))

    def get_sip_stats(self):
        return self.get_stats(self.sip_evaluation_joints)


class Logger:
    def __init__(self, filename, stdout=None):
        self.stdout = stdout
        self.logfile = open(filename, 'w')

    def print(self, text):
        if self.stdout is not None:
            self.stdout.write(text)
        self.logfile.write(text)

    def close(self):
        self.logfile.close()
