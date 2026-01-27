from functools import cache

import cv2
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist


def marker_lexsort(keypoints, tolerance: float = 5.0):
    """Perform an indirect stable sort on the keypoints so that the keypoints are arranged in rows.

    Args:
        keypoints (np.ndarray): _description_
        tolerance (float, optional): Tolerence in marker positions (px). Defaults to 5.0.
    """
    x_for_sort = np.round(keypoints[:, 0] / tolerance)
    y_for_sort = np.round(keypoints[:, 1] / tolerance)
    sort_indices = np.lexsort((x_for_sort, y_for_sort))
    return keypoints[sort_indices]


class FrameProcessor:
    def __init__(self):
        pass

    def process(self, frame: np.ndarray, ts: float = None) -> dict:
        return {}

    def reset(self, **kwargs) -> None:
        pass

    # @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> dict:
        return self.process(frame)

    @cache
    def get_output_dims(self) -> dict[str, int]:
        res = self.process(np.zeros((400, 400, 3), dtype=np.uint8))
        return {k: v.shape[-1] for k, v in res.items()}


def get_default_tacthru_tracker():
    det_params = cv2.SimpleBlobDetector_Params()
    det_params.filterByConvexity = False
    det_params.filterByColor = True
    det_params.blobColor = 0
    det_params.filterByArea = True
    det_params.minArea = 30
    det_params.maxArea = 400
    det_params.minDistBetweenBlobs = 0.5
    filter_double_det = cv2.SimpleBlobDetector_create(det_params)

    def double_det_fn(img: np.ndarray):
        res = {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thres = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO)
        thres = cv2.normalize(thres, None, 0, 255, cv2.NORM_MINMAX)

        res["thres"] = thres
        res["blobs"] = filter_double_det.detect(thres)
        return res

    return double_det_fn


class KeypointsKFProcessor(FrameProcessor):
    n_effective_markers: int = 0

    def __init__(self, ref_marker_pos, blob_det: callable = None):
        self.blob_det: callable = get_default_tacthru_tracker() if blob_det is None else blob_det
        ref_marker_pos = marker_lexsort(ref_marker_pos, tolerance=25)
        self.update_ref_markers(ref_marker_pos)

    def update_ref_markers(self, kpts_ref: np.ndarray):
        self.kpts_ref = kpts_ref
        self.kpts_ref_tree = KDTree(self.kpts_ref, metric="minkowski", p=2)
        self.kpts_buffer = np.zeros_like(self.kpts_ref)
        self.kpts_dist_buffer = np.zeros(len(self.kpts_ref))
        self.kpts_buffer_valid = np.zeros(len(self.kpts_ref), dtype=bool)
        self.marker_id = np.arange(len(self.kpts_ref))

        self.reset()

    def reset(self):
        self.n_kpts = 0
        self.max_n_kpts = len(self.kpts_ref)
        self.marker_arange = np.arange(self.max_n_kpts)

        self.x_max_dist = 25

        self.state_noise_cov = np.diag([0.105**2, 0.105**2])
        self.obs_noise_cov = np.diag([0.421**2, 0.421**2])
        self.kf_x = self.kpts_ref.copy()

        # Pre-allocate buffers for KF
        self.kf_z = self.kpts_ref.copy()
        self.kf_pred = self.kpts_ref.copy()

        self.kf_cov = np.zeros([len(self.kpts_ref), 2, 2], dtype=np.float32)

        self.kf_gain = np.zeros([len(self.kpts_ref), 2, 2], dtype=np.float32)
        self.kf_update = np.zeros_like(self.kf_x)
        self.kf_update_cov = np.zeros([len(self.kpts_ref), 2, 2], dtype=np.float32)

        self.S = np.zeros_like(self.kf_cov)
        self.S_inv = np.zeros_like(self.kf_cov)
        self.innovation = np.zeros_like(self.kf_x)
        self.identity = np.eye(2, dtype=np.float32)[None, :, :]
        self.tmp_cov = np.zeros_like(self.kf_cov)
        self.tmp_update = np.zeros((len(self.kpts_ref), 2, 1), dtype=np.float32)

    def process(self, frame):
        res = self.blob_det(frame)
        blobs = res.pop("blobs")
        all_kpts = cv2.KeyPoint_convert(blobs)

        kpts_to_x_cdist = cdist(self.kf_x, all_kpts)
        indices = np.argmin(kpts_to_x_cdist, axis=1)
        valid_mask = kpts_to_x_cdist[self.marker_arange, indices] < self.x_max_dist

        self.n_effective_markers = valid_mask.sum()

        # Prepare measurements
        matched_kpts = all_kpts[indices]
        np.copyto(self.kf_z, self.kf_x)
        self.kf_z[valid_mask] = matched_kpts[valid_mask]

        # Prediction: x_pred = x, P_pred = P + Q
        np.copyto(self.kf_pred, self.kf_x)
        self.kf_cov += self.state_noise_cov

        # S = P_pred + R
        np.add(self.kf_cov, self.obs_noise_cov, out=self.S)

        # Clean inversion
        det = self.S[:, 0, 0] * self.S[:, 1, 1] - self.S[:, 0, 1] * self.S[:, 1, 0]
        # Avoid zero div
        det = np.where(np.abs(det) < 1e-9, 1e-9, det)

        self.S_inv[:, 0, 0] = self.S[:, 1, 1] / det
        self.S_inv[:, 1, 1] = self.S[:, 0, 0] / det
        self.S_inv[:, 0, 1] = -self.S[:, 0, 1] / det
        self.S_inv[:, 1, 0] = -self.S[:, 1, 0] / det

        # K = P_pred @ S_inv
        np.matmul(self.kf_cov, self.S_inv, out=self.kf_gain)

        # for x_update calculation
        np.subtract(self.kf_z, self.kf_pred, out=self.innovation)
        np.matmul(self.kf_gain, self.innovation[..., None], out=self.tmp_update)
        np.add(self.kf_pred, self.tmp_update.squeeze(-1), out=self.kf_update)

        # P_upd = (I - K) P_pred
        np.subtract(self.identity, self.kf_gain, out=self.tmp_cov)
        np.matmul(self.tmp_cov, self.kf_cov, out=self.kf_update_cov)

        # Apply updates
        np.copyto(self.kf_x, self.kf_update)

        # For valid markers, use updated covariance.
        # For invalid markers, keep predicted covariance (which is already in self.kf_cov)
        self.kf_cov[valid_mask] = self.kf_update_cov[valid_mask]

        res.update({"marker": self.kf_x.copy(), "marker_ref": self.kpts_ref, "all_kpts": all_kpts})

        return res
