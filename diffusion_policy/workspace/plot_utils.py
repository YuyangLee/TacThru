import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from plotly import graph_objects as go

sns.set()


def plot_connection(x, y, color="white", name="conn"):
    return [
        go.Scatter3d(
            x=[x[i, 0], y[i, 0]],
            y=[x[i, 1], y[i, 1]],
            z=[x[i, 2], y[i, 2]],
            mode="lines",
            line={"color": color, "width": 2},
        )
        for i in range(x.shape[0])
    ]


def plot_point_cloud(pts, **kwargs):
    return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", **kwargs)


def plot_trajectory(trajectory: np.ndarray, pt_color="blue", line_color="green", name="trajectory", **kwargs):
    # trajectory: (T, 3); plot as a line with dots at the points
    assert trajectory.ndim == 2 and trajectory.shape[1] == 3, f"Invalid shape {trajectory.shape}"
    return [
        go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode="lines+markers",
            line=dict(color=line_color, width=4),
            marker=dict(size=5, color=pt_color),
            name=name,
            **kwargs,
        )
    ]


def plot_mesh(mesh, color="lightblue", opacity=1.0, name="mesh"):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        showlegend=True,
    )


def to_numpy(mat):
    if hasattr(mat, "cpu"):
        return mat.cpu().numpy()
    return np.array(mat)


def width_to_gripper_width_pos(poses, widths):
    local_from, local_to = np.zeros([len(widths), 3]), np.zeros([len(widths), 3])
    local_from[:, 0] = -widths / 2
    local_to[:, 0] = widths / 2
    from_pos = np.matmul(poses[:, :3, :3], local_from[..., None]).squeeze(-1) + poses[:, :3, 3]
    to_pos = np.matmul(poses[:, :3, :3], local_to[..., None]).squeeze(-1) + poses[:, :3, 3]
    gripper_width_pos = np.stack([from_pos, to_pos], axis=1)
    return gripper_width_pos  # (N, 2, 3)


def dump_frames(gopro_frames, tac_frames, filename):
    # Resize all_tac frame to match all_gopro frames' height, then hstack.
    for i_frame in range(len(gopro_frames)):
        gopro_height = gopro_frames[i_frame].shape[0]
        frame = gopro_frames[i_frame]
        if len(tac_frames) > 0:
            tac_height = tac_frames[i_frame].shape[0]
            resize_tac_width = int(tac_height * gopro_frames[i_frame].shape[1] / gopro_height)
            tac_frame = cv2.resize(tac_frames[i_frame], (resize_tac_width, gopro_height))
            frame = np.hstack((frame, tac_frame))
        frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

        if i_frame == 0:
            writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    writer.release()


def plot_ee_trajectories(gt_list, pred_list, hist_list):
    """
    Plots 3D end-effector trajectories for GT and Pred sequences with color gradients.

    Args:
        gt_list (List[Tuple[torch.Tensor/np.ndarray, torch.Tensor/np.ndarray]]):
            List of tuples, each containing:
            - (T, 4, 4) poses for ground truth
            - (T,) color values between 0 and 1
        pred_list (List[Tuple[torch.Tensor/np.ndarray, torch.Tensor/np.ndarray]]):
            List of tuples, each containing:
            - (T, 4, 4) poses for predictions
            - (T,) color values between 0 and 1

    Returns:
        fig (go.Figure): Plotly figure object.
    """
    fig = go.Figure()

    # plot GT
    for i, traj_tuple in enumerate(gt_list, start=1):
        poses, widths = map(to_numpy, traj_tuple)
        colors = np.linspace(0, 1, poses.shape[0])  # Use a linear gradient for colors

        # Plot all points with colors
        fig.add_trace(
            go.Scatter3d(
                x=poses[:, 0, 3],
                y=poses[:, 1, 3],
                z=poses[:, 2, 3],
                mode="markers+lines",
                name=f"GT - {i}",
                legendgroup=f"GT - {i}",
                marker=dict(size=2, color=colors, colorscale=[[0, "green"], [1, "white"]], showscale=False),
                line=dict(color="green", width=2),
            )
        )
        gripper_width_pos = width_to_gripper_width_pos(poses, widths)
        for i_width, width in enumerate(widths):
            fig.add_trace(
                go.Scatter3d(
                    x=gripper_width_pos[i_width, :, 0],
                    y=gripper_width_pos[i_width, :, 1],
                    z=gripper_width_pos[i_width, :, 2],
                    mode="markers+lines",
                    name=f"GT - {i}",
                    legendgroup=f"GT - {i}",
                    marker=dict(size=3, color="lightgreen", showscale=False),
                    line=dict(color="green", width=3),
                    showlegend=False,
                )
            )

    # plot Pred
    for i, traj_tuple in enumerate(pred_list, start=1):
        poses, widths = map(to_numpy, traj_tuple)
        colors = np.linspace(0, 1, poses.shape[0])  # Use a linear gradient for colors

        # Plot all points with colors
        fig.add_trace(
            go.Scatter3d(
                x=poses[:, 0, 3],
                y=poses[:, 1, 3],
                z=poses[:, 2, 3],
                mode="markers+lines",
                name=f"Pred - {i}",
                legendgroup=f"Pred - {i}",
                marker=dict(size=2, color=colors, colorscale=[[0, "red"], [1, "white"]], showscale=False),
                line=dict(color="red", width=2),
            )
        )

        gripper_width_pos = width_to_gripper_width_pos(poses, widths)
        for i_width, width in enumerate(widths):
            fig.add_trace(
                go.Scatter3d(
                    x=gripper_width_pos[i_width, :, 0],
                    y=gripper_width_pos[i_width, :, 1],
                    z=gripper_width_pos[i_width, :, 2],
                    mode="markers+lines",
                    name=f"Pred - {i}",
                    legendgroup=f"Pred - {i}",
                    marker=dict(size=3, color="lightpink", showscale=False),
                    line=dict(color="red", width=3),
                    showlegend=False,
                )
            )

    # plot Hist
    for i, traj_tuple in enumerate(hist_list, start=1):
        poses, widths = map(to_numpy, traj_tuple)
        colors = np.linspace(0, 1, poses.shape[0])  # Use a linear gradient for colors

        # Plot all points with colors
        fig.add_trace(
            go.Scatter3d(
                x=poses[:, 0, 3],
                y=poses[:, 1, 3],
                z=poses[:, 2, 3],
                mode="markers+lines",
                name=f"Hist - {i}",
                legendgroup=f"Hist - {i}",
                marker=dict(size=2, color=colors, colorscale=[[0, "blue"], [1, "green"]], showscale=False),
                line=dict(color="blue", width=2),
            )
        )

        gripper_width_pos = width_to_gripper_width_pos(poses, widths)
        for i_width, width in enumerate(widths):
            fig.add_trace(
                go.Scatter3d(
                    x=gripper_width_pos[i_width, :, 0],
                    y=gripper_width_pos[i_width, :, 1],
                    z=gripper_width_pos[i_width, :, 2],
                    mode="markers+lines",
                    name=f"Pred - {i}",
                    legendgroup=f"Pred - {i}",
                    marker=dict(size=3, color="lightpink", showscale=False),
                    line=dict(color="blue", width=3),
                    showlegend=False,
                )
            )

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        legend=dict(itemsizing="constant"),
    )

    return fig
