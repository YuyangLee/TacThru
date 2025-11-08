import random
from typing import Optional

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
from loguru import logger

from diffusion_policy.common.replay_buffer import ReplayBuffer


def get_val_mask(n_episodes, val_ratio, seed=0):
    rng = np.random.default_rng(seed=seed)
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    demo_indices = np.arange(n_episodes)
    rng.shuffle(demo_indices)
    val_indices = demo_indices[:n_val]
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    val_mask[val_indices] = True
    logger.info(f"# of episodes (train / eval / all): {(~val_mask).sum()} / {val_mask.sum()} / {n_episodes}")
    return val_mask


class SequenceSampler:
    def __init__(
        self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,
        tacthru_keys: list,
        lowdim_keys: list,
        key_horizon: dict,
        key_latency_steps: dict,
        key_down_sample_steps: dict,
        episode_mask: np.ndarray | None = None,
        action_padding: bool = False,
        repeat_frame_prob: float = 0.0,
        max_duration: float | None = None,
        curr_index_interval: int = 1,
    ):
        episode_ends = replay_buffer.episode_ends[:]

        # load gripper_width
        gripper_width = replay_buffer["robot0_gripper_width"][:, 0]
        gripper_width_threshold = 0.075
        self.repeat_frame_prob = repeat_frame_prob

        # create indices, including (current_idx, start_idx, end_idx)
        self.indices = list()
        for episode_idx in range(len(episode_ends)):
            before_first_grasp = True  # initialize for each episode
            if episode_mask is not None and not episode_mask[episode_idx]:  # skip episodes not for validation
                continue
            start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
            end_idx = episode_ends[episode_idx]
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)

            for current_idx in range(start_idx, end_idx, curr_index_interval):
                if (
                    not action_padding
                    and end_idx < current_idx + (key_horizon["action"] - 1) * key_down_sample_steps["action"] + 1
                ):
                    continue
                if gripper_width[current_idx] < gripper_width_threshold:
                    before_first_grasp = False
                self.indices.append((episode_idx, current_idx, start_idx, end_idx, before_first_grasp))

        # load low_dim to memory and keep rgb as compressed zarr array
        self.replay_buffer = dict()
        self.num_robot = 0
        for key in lowdim_keys:
            if key.endswith("eef_pos"):
                self.num_robot += 1

            if key.endswith("pos_abs"):
                axis = shape_meta["obs"][key]["axis"]
                if isinstance(axis, int):
                    axis = [axis]
                self.replay_buffer[key] = replay_buffer[key[:-4]][:, list(axis)]
            elif key.endswith("quat_abs"):
                axis = shape_meta["obs"][key]["axis"]
                if isinstance(axis, int):
                    axis = [axis]
                # HACK for hybrid abs/relative proprioception
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_quat(rot_in).as_euler("XYZ")
                self.replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith("axis_angle_abs"):
                axis = shape_meta["obs"][key]["axis"]
                if isinstance(axis, int):
                    axis = [axis]
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_rotvec(rot_in).as_euler("XYZ")
                self.replay_buffer[key] = rot_out[:, list(axis)]
            else:
                self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys + tacthru_keys:
            self.replay_buffer[key] = replay_buffer[key]

        if "action" in replay_buffer:
            self.replay_buffer["action"] = replay_buffer["action"][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ["eef_pos", "eef_rot_axis_angle", "gripper_width"]:
                    key = f"robot{robot_idx}_{cat}"
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer["action"] = np.concatenate(actions, axis=-1)

        self.action_padding = action_padding
        self.rgb_keys = rgb_keys
        self.tacthru_keys = tacthru_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps

        self.ignore_rgb_is_applied = False  # speed up the interation when getting normalizaer

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        episode_idx, current_idx, start_idx, end_idx, before_first_grasp = self.indices[idx]

        result = dict()

        """
        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys
        """

        obs_keys = self.lowdim_keys
        if not self.ignore_rgb_is_applied:
            obs_keys = self.tacthru_keys + self.rgb_keys + obs_keys

        # observation
        for key in obs_keys:
            input_arr = self.replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_latency_steps = self.key_latency_steps[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            if key in self.rgb_keys or key in self.tacthru_keys:
                assert this_latency_steps == 0
                num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
                slice_start = current_idx - (num_valid - 1) * this_downsample_steps

                output = input_arr[slice_start : current_idx + 1 : this_downsample_steps]
                assert output.shape[0] == num_valid

                # solve padding
                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
            else:
                idx_with_latency = np.array(
                    [current_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
                    dtype=np.float32,
                )
                idx_with_latency = idx_with_latency[::-1]
                idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                if "rot" in key:
                    # rotation
                    rot_preprocess, rot_postprocess = None, None
                    if key.endswith("quat"):
                        rot_preprocess = st.Rotation.from_quat
                        rot_postprocess = st.Rotation.as_quat
                    elif key.endswith("axis_angle"):
                        rot_preprocess = st.Rotation.from_rotvec
                        rot_postprocess = st.Rotation.as_rotvec
                    else:
                        raise NotImplementedError
                    slerp = st.Slerp(
                        times=np.arange(interpolation_start, interpolation_end),
                        rotations=rot_preprocess(input_arr[interpolation_start:interpolation_end]),
                    )
                    output = rot_postprocess(slerp(idx_with_latency))
                else:
                    interp = si.interp1d(
                        x=np.arange(interpolation_start, interpolation_end),
                        y=input_arr[interpolation_start:interpolation_end],
                        axis=0,
                        assume_sorted=True,
                    )
                    output = interp(idx_with_latency)

            result[key] = output

        # repeat frame before first grasp
        if self.repeat_frame_prob != 0.0:
            if before_first_grasp and random.random() < self.repeat_frame_prob:
                for key in obs_keys:
                    result[key][:-1] = result[key][-1:]

        # aciton
        input_arr = self.replay_buffer["action"]
        action_horizon = self.key_horizon["action"]
        action_latency_steps = self.key_latency_steps["action"]
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps["action"]
        slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_idx:slice_end:action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result["action"] = output
        result["episode_idx"] = episode_idx
        result["episode_t"] = current_idx - start_idx

        return result

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply
