# TacThru: Fingetip-integrated Tactile and Visual Perception for Fine-grained and Contact-rich Manipulation

**Yuyang Li <sup>1,2,3,4\*</sup>, Yinghan Chen <sup>1,2,4,6\*</sup>, Zihang Zhao <sup>1,2,4</sup>, Puhao Li <sup>3,4</sup>, Tengyu Liu <sup>3,4&dagger;</sup>, Siyuan Huang <sup>3,4&dagger;</sup>, and Yixin Zhu <sup>1,2,4,5&dagger;</sup>**

<sup>\*</sup> Equal contribution&nbsp;&nbsp;<sup>&dagger;</sup> Corresponding Authors

<sup>1</sup> Peking University<br/>
<sup>2</sup> Beijing Key Lab of Behavior and Mental Health, Peking University<br/>
<sup>3</sup> Beijing Institute for General Artificial Intelligence<br/>
<sup>4</sup> State Key Lab of General Artificial Intelligence<br/>
<sup>5</sup> PKU-Wuhan Institute for Artificial Intelligence<br/>
<sup>6</sup> University of Cambridge<br/>

[🌐 Website](https://go.yuyang.li/tacthru) |
[📑 Paper](./assets/tacthru-paper.pdf) |
[📹 Video](https://vimeo.com) |
[💾 Datasets](https://huggingface.co/datasets/aidenli/tacthru_umi_tasks) |
[🛠️ Hardware Guide](https://docs.google.com/document/d/1fpZRiGoxWqLoFs-zxnG4d_d3hy0eHjlLA4nsuEKvCEg/edit?usp=sharing)

## Codebase Structure

- `assets/` includes necesssary assets for this repo.
- `cfg/` holds the configs for policy training.
- `data/`
    - `data/marker_tests/` includes data for testing keyline marker tracking. It will also contain the downloaded TacThr-UMI datasets.
    - `data/tacthru/` includes the example model of our sensor, and the STEP file for our TacThru-UMI gripper.
- `diffusion_policy/` includes necessary utils for the Diffusion Policy. The codes are modified from [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).
- `utils/` hold utilities for TacThru signal processing and policy learning.
- `scripts/` includes scripts for starting marker test, training, etc.

## Environment Setup

Clone the repo:

```shell
git clone https://github.com/YuyangLee/TacThru
cd TacThru
```

We use `uv` to manage the virtual environment:

```shell
uv sync
```

By default, the dependencies include necessary tools to test the keyline marker tracking of TacThru. Optional dependencies are used fortraining and validating robotic manipulation policies with TacThru-UMI:

```shell
uv sync --extra umi
```

## Marker Test

We provide two examples of our marker tracking algorithms:

```shell
uv run scripts/marker_test.py
```

---

## Download Datasets

We provide all the datasets used in our experiments:

- `PickBottle`
- `PullTissue`
- `SortBolt`
- `HangScissors`
- `InsertCap`

They are provided in [our Hugging Face Dataset](https://huggingface.co/datasets/aidenli/tacthru_umi_tasks) and set up as a sub module under `data/tasks/` in this repo. Make sure you have access to the Hugging Face public datasets. To sync them:

```
git submodule init
git submodule update
```

You can also use [sprse checkout](https://git-scm.com/docs/git-sparse-checkout) to download partially.

## Train Policy

[`./train_tf.sh`](./train_tf.sh) shows an example training script:

```shell
task=pick_bottle

# TacThru w/ marker deviations
tac_active_keys="[tacthru_l_rgb,tacthru_l_markers]"
obs_tag="tt_m"
exp_tag="run"

uv run scripts/train.py --config-name=train_tf exp_name=tf-$obs_tag-$exp_tag task=$task tac_active_keys=$tac_active_keys
```

The `task` is one of: `pick_bottle`, `pull_tissue`, `sort_bolt`, `hang_scissors`, `insert_cap`.

The `tac_active_keys` must includes the items in `train.task.shape_meta.obs`. In the provided dataset, `tacthru_l_*` belongs to the TacThru signals, installed as the left finger. `tacthru_r_*` belongs to the GelSight-type sensor signals (rectified).

### Add New Tasks

While we use a customized sensor, this codebase is theoretically compatible with any UMI system with vision-based tactile or STS sensor.

#### Dataset Structure

First, prepare a Zarr file for your dataset. The structure is as follows:

```
data/camera0_rgb               shape=(108922, 224, 224, 3) dtype=uint8
data/robot0_demo_end_pose      shape=(108922, 6) dtype=float64
data/robot0_demo_start_pose    shape=(108922, 6) dtype=float64
data/robot0_eef_pos            shape=(108922, 3) dtype=float32
data/robot0_eef_rot_axis_angle shape=(108922, 3) dtype=float32
data/robot0_gripper_width      shape=(108922, 1) dtype=float32
data/tacthru_l_marker          shape=(108922, 64, 2) dtype=float32
data/tacthru_l_rgb             shape=(108922, 224, 224, 3) dtype=uint8
data/tacthru_r_marker          shape=(108922, 64, 2) dtype=float32
data/tacthru_r_rgb             shape=(108922, 224, 224, 3) dtype=uint8
meta/episode_ends              shape=(147,) dtype=int64
```

You can use `scripts/show_ds.py` to inspect the given Zarr file for reference.

#### Training

After getting the dataset Zarr file ready, create a task config in `cfg/train/task/`, with `_template.yaml` as a reference. A few things to modify:

- `name`: Your task name
- `dataset.dataset_path`: The file path of your Zarr file

## Hardware Guide

You can check out hardware guide in [🛠️ Hardware Guide](https://docs.google.com/document/d/1fpZRiGoxWqLoFs-zxnG4d_d3hy0eHjlLA4nsuEKvCEg/edit?usp=sharing).

![UMI Robotic Gripper](./assets/umi_asm.png)

## More Information

If you find our work helpful, please consider citing it:

```bibtex
{
    // TBD
}
```
