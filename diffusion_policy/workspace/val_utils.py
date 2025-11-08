import os
from io import BytesIO
from time import time

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tqdm
from loguru import logger
from PIL import Image

from diffusion_policy.common.pose_util import pose9d_to_mat_torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.plot_utils import dump_frames, plot_ee_trajectories
from utils import math_utils

# Set font size 20 pt
sns.set()
plt.rcParams.update({"font.size": 20})


def plot_mean_and_ci(traj, ylabel: str, title: str):
    B, T = traj.shape
    time_steps = range(T)

    mean_diff = torch.mean(traj, dim=0).cpu().numpy()
    std_diff = torch.std(traj, dim=0).cpu().numpy()
    plt.fill_between(
        time_steps,
        mean_diff - 1.96 * std_diff,
        mean_diff + 1.96 * std_diff,
        alpha=0.2,
        label="95% CI",
    )
    plt.plot(
        time_steps,
        mean_diff,
        label="Mean",
        linestyle="solid",
    )
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()


def plot_horizon_diff(gt_action, pred_action, out_path, epoch, tag: str = "Train"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    B, T, _ = gt_action.shape

    plt.figure(figsize=(10, 10))

    # 1. Position
    plt.subplot(3, 1, 1)
    position_diff = torch.norm(gt_action[:, :, :3] - pred_action[:, :, :3], p=2, dim=-1)  # B x T
    plot_mean_and_ci(
        position_diff,
        "Position Diff (m)",
        f"Position Comparison (Epoch {epoch}) - {tag}",
    )
    plt.grid()

    # 2. Rotation
    plt.subplot(3, 1, 2)
    gt_d6 = gt_action[:, :, 3:9]
    pred_d6 = pred_action[:, :, 3:9]
    axis_angle_diff = compute_axis_angle_difference(gt_d6.reshape(-1, 6), pred_d6.reshape(-1, 6)).reshape(B, T, 3)
    axis_angle_diff_norm = torch.norm(axis_angle_diff, p=2, dim=-1) / torch.pi * 180
    plot_mean_and_ci(
        axis_angle_diff_norm,
        "Rotation Diff (deg)",
        f"Rotation Comparison (Epoch {epoch}) - {tag}",
    )
    plt.grid()

    # 3. Width
    plt.subplot(3, 1, 3)
    width_diff = gt_action[:, :, 9] - pred_action[:, :, 9]
    plot_mean_and_ci(width_diff, "Width Diff", f"Width Comparison (Epoch {epoch}) - {tag}")
    plt.grid()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    matrix = math_utils.rotation_6d_to_matrix(d6)
    axis_angle = math_utils.axis_angle_from_matrix(matrix)
    return axis_angle


def compute_axis_angle_difference(gt_d6, pred_d6):
    # logger.info(f"gt_d6 shape: {gt_d6.shape}, pred_d6 shape: {pred_d6.shape}")
    gt_axis_angle = rotation_6d_to_axis_angle(gt_d6)
    pred_axis_angle = rotation_6d_to_axis_angle(pred_d6)
    # logger.info(f"gt_axis_angle shape: {gt_axis_angle.shape}, pred_axis_angle shape: {pred_axis_angle.shape}")
    diff = gt_axis_angle - pred_axis_angle
    # logger.info(f"axis_angle_diff shape: {diff.shape}")
    return diff


def save_trajectory_csv(position: torch.Tensor, output_path: str):
    if isinstance(position, torch.Tensor):
        position = position.detach().cpu().numpy()
    elif not isinstance(position, np.ndarray):
        raise TypeError("Expected torch.Tensor or np.ndarray")
    T = position.shape[0]
    csv_data = np.zeros((T, 4))  # timestep, x, y, z
    csv_data[:, 0] = np.arange(T)
    csv_data[:, 1:] = position[:, :3]
    header = "timestep,x,y,z"
    np.savetxt(output_path, csv_data, delimiter=",", header=header, comments="", fmt="%.6f")


def eval_action_l1(category, pred_action, gt_action):
    eval_step_log = {}
    B, T, _ = pred_action.shape
    pred_action = pred_action.view(B, T, -1, 10)
    gt_action = gt_action.view(B, T, -1, 10)
    eval_step_log[f"{category}_action_l1_error"] = torch.nn.functional.l1_loss(pred_action, gt_action).item()
    eval_step_log[f"{category}_action_l1_error_pos"] = torch.nn.functional.l1_loss(
        pred_action[..., :3], gt_action[..., :3]
    ).item()
    eval_step_log[f"{category}_action_l1_error_rot"] = torch.nn.functional.l1_loss(
        pred_action[..., 3:9], gt_action[..., 3:9]
    ).item()
    eval_step_log[f"{category}_action_l1_error_width"] = torch.nn.functional.l1_loss(
        pred_action[..., 9], gt_action[..., 9]
    ).item()
    return eval_step_log


def overlay_attn(image: np.ndarray, weights: np.ndarray, normalize: bool = True) -> np.ndarray:
    cmap = plt.get_cmap("viridis")
    weights_viz = weights / np.max(weights) if normalize else weights  # Normalize to [0, 1]
    weight_colors = cmap(weights_viz)[:, :, 2::-1].astype(np.float32)  # Get RGB colors from colormap
    weight_colors = cv2.resize(weight_colors, (image.shape[1], image.shape[0]))  # Resize to match image size
    weight_colors = (weight_colors * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    overlayed_image = cv2.addWeighted(image, 0.4, weight_colors, 0.6, 0)  # Blend the images

    n_tokens_row, n_tokens_col = weights_viz.shape
    block_h = image.shape[0] // n_tokens_row
    block_w = image.shape[1] // n_tokens_col

    for i in range(n_tokens_row):
        for j in range(n_tokens_col):
            text = str(int(weights_viz[i, j] * 100))
            text_position = (j * block_w + 2, i * block_h + 8)
            cv2.putText(
                overlayed_image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0, 0.5), 1, cv2.LINE_AA
            )
            if i < n_tokens_row - 1 and j < n_tokens_col - 1:
                cv2.drawMarker(
                    overlayed_image,
                    ((j + 1) * block_w - 1, (i + 1) * block_h - 1),
                    color=(0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=8,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )
    return overlayed_image


def plot_attention_bars(attention_weights: np.ndarray, token_labels_list: list[str]) -> np.ndarray:
    """
    Generates a horizontal bar plot image as a NumPy array from attention weights.
    Each token is colored individually, weights are averaged over the T dimension.

    Args:
        attention_weights (np.ndarray): A T x N_token NumPy array of attention weights.
        token_labels_list (list[str]): A list of N_token strings, where consecutive identical
                                       strings form segments.

    Returns:
        np.ndarray: A NumPy array representing the RGB image of the plot.
    """
    T, N_token = attention_weights.shape

    # Average weights over T dimension
    avg_attention_weights = np.mean(attention_weights, axis=0)  # Shape: (N_token,)

    segments = []
    if N_token > 0:
        if not token_labels_list:
            raise ValueError("token_labels_list is empty but N_token > 0")

        current_label = token_labels_list[0]
        start_index = 0
        for i in range(1, N_token):
            if token_labels_list[i] != current_label:
                segments.append({"label": current_label, "start": start_index, "end": i, "count": i - start_index})
                current_label = token_labels_list[i]
                start_index = i
        segments.append({"label": current_label, "start": start_index, "end": N_token, "count": N_token - start_index})

    # Single subplot configuration
    figure_height_inches = 2.0
    figure_width_inches = 7

    if N_token == 0:
        fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches))
        ax.text(0.5, 0.5, "No tokens to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches))

    cmap = plt.cm.viridis

    if N_token > 0:
        ax.text(-0.03, 0.5, f"Avg", transform=ax.transAxes, ha="right", va="center", fontsize=9, color="gray")
        ax.get_yaxis().set_visible(False)
        ax.set_ylim(-0.5, 0.5)

        min_val = np.min(avg_attention_weights)
        max_val = np.max(avg_attention_weights)

        if min_val == max_val:
            if min_val == 0:
                norm = mcolors.Normalize(vmin=0, vmax=1.0)
            else:
                delta = abs(min_val * 0.1) if min_val != 0 else 0.1
                norm = mcolors.Normalize(vmin=min_val - delta, vmax=max_val + delta)
        else:
            norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        for seg_info in segments:
            label = seg_info["label"]
            start_idx = seg_info["start"]
            end_idx = seg_info["end"]
            count = seg_info["count"]

            if count == 0:
                continue

            for token_idx in range(start_idx, end_idx):
                token_color = scalar_mappable.to_rgba(avg_attention_weights[token_idx])
                ax.barh(0, 1, left=token_idx, color=token_color, height=0.7, edgecolor="none")

            avg_segment_att_for_text_color = np.mean(avg_attention_weights[start_idx:end_idx])
            avg_bar_color_for_text = scalar_mappable.to_rgba(avg_segment_att_for_text_color)
            r_text, g_text, b_text, _ = avg_bar_color_for_text
            luminance = 0.299 * r_text + 0.587 * g_text + 0.114 * b_text
            text_color = "white" if luminance < 0.45 else "black"
            font_size = 9
            if count > len(label) * 0.5 or count > N_token * 0.05:
                ax.text(
                    start_idx + count / 2,
                    0,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=font_size,
                    fontweight="normal",
                )

        # Add vertical separator lines between segments
        for i, seg_info in enumerate(segments):
            segment_end_pos = seg_info["end"]
            if i < len(segments) - 1:  # Don't draw a line after the last segment
                ax.vlines(x=segment_end_pos, ymin=-0.35, ymax=0.35, color="black", linewidth=0.8, linestyle="-")

        ax.set_xlim(0, N_token)

        if N_token <= 10:
            tick_step = 1
        elif N_token <= 50:
            tick_step = 5
        else:
            tick_step = max(1, N_token // 10)

        xticks = np.arange(0, N_token + 1, tick_step)
        if N_token > 0 and N_token not in xticks:
            xticks = np.unique(np.append(xticks, N_token))

        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelsize=8, pad=2)

        # Add colorbar
        cax = ax.inset_axes([1.02, 0.1, 0.03, 0.8])
        cbar = fig.colorbar(scalar_mappable, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)

    fig.subplots_adjust(left=0.08, right=0.90, top=0.95, bottom=0.15)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    image_array = np.array(Image.open(buf))
    plt.close(fig)
    buf.close()

    return image_array[..., :3]


def plot_attention_bars_T(attention_weights: np.ndarray, token_labels_list: list[str]) -> np.ndarray:
    """
    Generates a stacked horizontal bar plot image as a NumPy array from attention weights.
    Each token is colored individually, each row has ticks and its own colorbar,
    and vertical bars separate segments.

    Args:
        attention_weights (np.ndarray): A T x N_token NumPy array of attention weights.
        token_labels_list (list[str]): A list of N_token strings, where consecutive identical
                                       strings form segments.

    Returns:
        np.ndarray: A NumPy array representing the RGBA image of the plot.
    """
    T, N_token = attention_weights.shape

    segments = []
    if N_token > 0:
        if not token_labels_list:
            raise ValueError("token_labels_list is empty but N_token > 0")

        current_label = token_labels_list[0]
        start_index = 0
        for i in range(1, N_token):
            if token_labels_list[i] != current_label:
                segments.append({"label": current_label, "start": start_index, "end": i, "count": i - start_index})
                current_label = token_labels_list[i]
                start_index = i
        segments.append({"label": current_label, "start": start_index, "end": N_token, "count": N_token - start_index})

    subplot_height_inches = 0.4
    figure_height_inches = max(1.5, T * subplot_height_inches if T > 0 else subplot_height_inches)
    figure_width_inches = 7

    if T == 0:
        fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches))
        ax.text(0.5, 0.5, "No data to plot (T=0)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        fig, axes = plt.subplots(T, 1, figsize=(figure_width_inches, figure_height_inches), squeeze=False, sharex=True)
        axes = axes.flatten()

    cmap = plt.cm.viridis

    # The code below takes ~10 secs!
    for t_idx in range(T):
        ax = axes[t_idx]
        current_attn_row = attention_weights[t_idx, :]

        ax.text(-0.03, 0.5, f"T{t_idx}", transform=ax.transAxes, ha="right", va="center", fontsize=7, color="gray")
        ax.get_yaxis().set_visible(False)
        ax.set_ylim(-0.5, 0.5)

        if N_token == 0:
            ax.text(0.5, 0.5, "N_token = 0", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.get_xaxis().set_visible(False)
            continue

        min_val = np.min(current_attn_row)
        max_val = np.max(current_attn_row)

        if min_val == max_val:
            if min_val == 0:
                norm = mcolors.Normalize(vmin=0, vmax=1.0)
            else:
                delta = abs(min_val * 0.1) if min_val != 0 else 0.1
                norm = mcolors.Normalize(vmin=min_val - delta, vmax=max_val + delta)
        else:
            norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        for seg_info in segments:
            label = seg_info["label"]
            start_idx = seg_info["start"]
            end_idx = seg_info["end"]
            count = seg_info["count"]

            if count == 0:
                continue

            for token_idx in range(start_idx, end_idx):
                token_color = scalar_mappable.to_rgba(current_attn_row[token_idx])
                ax.barh(0, 1, left=token_idx, color=token_color, height=0.7, edgecolor="none")

            avg_segment_att_for_text_color = np.mean(current_attn_row[start_idx:end_idx])
            avg_bar_color_for_text = scalar_mappable.to_rgba(avg_segment_att_for_text_color)
            r_text, g_text, b_text, _ = avg_bar_color_for_text
            luminance = 0.299 * r_text + 0.587 * g_text + 0.114 * b_text
            text_color = "white" if luminance < 0.45 else "black"
            font_size = 7
            if count > len(label) * 0.5 or count > N_token * 0.05:
                ax.text(
                    start_idx + count / 2,
                    0,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=font_size,
                    fontweight="normal",
                )

        # Add vertical separator lines between segments
        if N_token > 0:
            segment_end_pos = 0
            for i, seg_info in enumerate(segments):
                segment_end_pos = seg_info["end"]  # This is the boundary after the current segment
                if i < len(segments) - 1:  # Don't draw a line after the last segment
                    ax.vlines(x=segment_end_pos, ymin=-0.35, ymax=0.35, color="black", linewidth=0.8, linestyle="-")

        ax.set_xlim(0, N_token)

        if N_token <= 10:
            tick_step = 1
        elif N_token <= 50:
            tick_step = 5
        else:
            tick_step = max(1, N_token // 10)

        xticks = np.arange(0, N_token + 1, tick_step)  # Ensure last tick is N_token or beyond
        if N_token > 0 and N_token not in xticks:
            xticks = np.unique(np.append(xticks, N_token))

        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelsize=6, pad=1)

        if T > 1 and t_idx < T - 1:
            ax.tick_params(axis="x", labelbottom=False)

        cax = ax.inset_axes([1.02, 0.05, 0.015, 0.9])
        cbar = fig.colorbar(scalar_mappable, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_visible(False)

    if T > 0:
        fig.subplots_adjust(left=0.05, right=0.90, top=0.98, bottom=0.08, hspace=0.3)
    else:
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)  # Reverted to user's preferred DPI
    buf.seek(0)
    image_array = np.array(Image.open(buf))
    plt.close(fig)
    buf.close()

    return image_array[..., :3]


N_val_times = 0


@torch.no_grad()
def run_validation(
    model, policy, val_ds, val_dl, epoch: int, output_dir: str, drop_history: list[str] = [], debug: bool = False
):
    val_step_log = {}
    val_losses = []
    all_obs_traj, all_gt_traj, all_pred_traj = [], [], []
    all_gopro, all_tac = [], []

    base_ee_tf = torch.eye(4, device=model.device).unsqueeze(0)

    curr_episode_idx, next_ep_t = None, 0

    with tqdm.tqdm(val_dl, desc=f"Validation epoch {epoch}", leave=False) as tepoch:
        save_dir = os.path.join(output_dir, "val")
        for batch_idx, batch in enumerate(tepoch):
            metadata = batch["metadata"]
            batch = dict_apply(batch, lambda x: x.to(policy.device, non_blocking=True))
            loss = model(batch)
            val_losses.append(loss)

            for k in batch["obs"].keys():
                if k in drop_history:
                    batch["obs"][k][:, :-1] = batch["obs"][k][:, -1:]

            gt_action = batch["action"]
            pred_outputs = policy.predict_action(batch["obs"], need_attn_weight=True, no_randomize=True)
            pred_action = pred_outputs["action_pred"]
            token_ids, attn_weights = (pred_outputs.get("token_ids"), pred_outputs.get("attn_weights"))
            if attn_weights is not None:
                attn_weights = attn_weights.mean(dim=-1).cpu().numpy()  # B x T x (N_tokens + 1)
            # Cumulate open-loop inference trajectory
            for idx, (ep_i, ep_t) in enumerate(zip(metadata["episode_idx"].tolist(), metadata["episode_t"].tolist())):
                if ep_i != curr_episode_idx:
                    # Moving into next episode
                    if curr_episode_idx is not None:
                        # Save the previous episode's data
                        os.makedirs(save_dir, exist_ok=True)

                        fig = plot_ee_trajectories(all_gt_traj, all_pred_traj, all_obs_traj)
                        fig.write_html(os.path.join(save_dir, f"epoch_{epoch}-{curr_episode_idx}.html"))

                        if epoch == 0:
                            dump_path = os.path.join(save_dir, f"epoch_{epoch}-frames.mp4")
                            dump_frames(all_gopro, all_tac, dump_path)

                        all_obs_traj, all_gt_traj, all_pred_traj = [], [], []
                        all_gopro, all_tac = [], []
                        next_ep_t = 0
                        base_ee_tf = torch.eye(4, device=model.device).unsqueeze(0)
                    curr_episode_idx = ep_i

                if "camera0_rgb" in batch["obs"]:
                    frame = batch["obs"]["camera0_rgb"][idx, -1].cpu().permute(1, 2, 0).numpy()
                    frame = (frame * 255).astype(np.uint8)
                    all_gopro.append(frame)
                    T = batch["obs"]["camera0_rgb"].shape[1]
                for side in "lr":
                    rgb_key_name = f"tacthru_{side}_rgb"
                    if rgb_key_name not in batch["obs"]:
                        continue
                    T = batch["obs"][rgb_key_name].shape[1]
                    tac_frame = batch["obs"][rgb_key_name][idx, -1].cpu().permute(1, 2, 0).numpy()
                    tac_frame = (tac_frame * 255).astype(np.uint8).copy()
                    all_tac.append(tac_frame)
                if ep_t == next_ep_t:
                    # Save the current episode next inference
                    obs_ee_pose = torch.concat(
                        [batch["obs"]["robot0_eef_pos"][idx], batch["obs"]["robot0_eef_rot_axis_angle"][idx]], dim=-1
                    )
                    obs_ee_tf = base_ee_tf @ pose9d_to_mat_torch(obs_ee_pose)  # T_obs x 4 x 4
                    obs_ee_width = batch["obs"]["robot0_gripper_width"][idx, :, 0]
                    gt_ee_tf = base_ee_tf @ pose9d_to_mat_torch(gt_action[idx, :, :9])
                    gt_ee_width = gt_action[idx, :, 9]
                    pred_ee_tf = base_ee_tf @ pose9d_to_mat_torch(pred_action[idx, :, :9])
                    pred_ee_width = pred_action[idx, :, 9]
                    all_obs_traj.append((obs_ee_tf, obs_ee_width))
                    all_gt_traj.append((gt_ee_tf, gt_ee_width))  # T x 4 x 4
                    all_pred_traj.append((pred_ee_tf, pred_ee_width))  # T x 4 x 4

                    base_ee_tf[:] = gt_ee_tf[-1]
                    next_ep_t += policy.action_horizon * val_ds.key_down_sample_steps["action"]

                    if attn_weights is not None:
                        inst_attn_weights = attn_weights[idx].mean(axis=0)  # (N_tokens + 1)
                        inst_attn_weights = inst_attn_weights / (inst_attn_weights.sum() + 1e-6)
                        # Plot overall attention bars
                        bars_array = plot_attention_bars(attn_weights[idx], token_ids)

                        # Plot RGB channels and overlay attention weights for wristcam and sensor RGB images
                        all_tokens = list(set(token_ids))
                        frame_and_maps = []
                        for token in all_tokens:
                            if not any(token.startswith(prefix) for prefix in ["camera", "sensor"]):
                                continue

                            frame = all_gopro[-1] if token.startswith("camera") else all_tac[-1]
                            frame = cv2.resize(frame, (256, 256))
                            token_idxs = [i for i, t in enumerate(token_ids) if t == token]
                            frame_attn_weights = inst_attn_weights[token_idxs].reshape([T, -1])[-1]  # , 1:]
                            n_side = int(np.sqrt(len(frame_attn_weights)))
                            frame_attn_map = overlay_attn(frame, frame_attn_weights.reshape(n_side, n_side), True)
                            frame_and_maps.append(np.vstack((frame, frame_attn_map)))

                        frame_and_maps = np.hstack(frame_and_maps)

                        target_width = bars_array.shape[1]
                        resize_frame_height = int(frame_and_maps.shape[0] * target_width / frame_and_maps.shape[1])
                        frame_and_maps = cv2.resize(frame_and_maps, (target_width, resize_frame_height))
                        frame_and_maps = np.vstack((frame_and_maps, bars_array))
                        export_path = os.path.join(save_dir, f"epoch_{epoch}-{curr_episode_idx}-step_{ep_t}.png")
                        os.makedirs(os.path.dirname(export_path), exist_ok=True)
                        cv2.imwrite(export_path, frame_and_maps[..., ::-1])
                        logger.info(f"Saved overlayed attention frame to {export_path}")

            if batch_idx == 0:
                save_path = os.path.join(output_dir, "val", f"epoch_{epoch}-val_horizon_diff.png")
                plot_horizon_diff(gt_action, pred_action, out_path=save_path, epoch=epoch, tag="val")
                eval_step_log = eval_action_l1("val", pred_action, gt_action)
                val_step_log.update(eval_step_log)

            if debug:
                break

    if len(val_losses) > 0:
        val_loss = torch.mean(torch.tensor(val_losses)).item()
        # log epoch average validation loss
        val_step_log["val_loss"] = val_loss

    return val_step_log


@torch.no_grad()
def stat_last_train_batch(policy, train_sampling_batch, epoch, output_dir):
    sample_log = {}
    # sample trajectory from training set, and evaluate difference
    batch = dict_apply(train_sampling_batch, lambda x: x.to(policy.device, non_blocking=True))
    gt_action = batch["action"]
    pred_action = policy.predict_action(batch["obs"])["action_pred"]
    eval_step_log = eval_action_l1("train", pred_action, gt_action)
    sample_log.update(eval_step_log)

    save_dir = os.path.join(output_dir, "val", f"epoch_{epoch}-train_horizon_diff.png")
    plot_horizon_diff(gt_action=gt_action, pred_action=pred_action, out_path=save_dir, epoch=epoch, tag="train")

    del batch
    del gt_action
    del pred_action
    return sample_log
