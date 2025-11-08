from functools import cache

import cv2
import numpy as np
from sklearn.neighbors import KDTree


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
    def __call__(self, frame: np.ndarray, ts: float = None) -> dict:
        return self.process(frame, ts)

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
    def __init__(self, ref_marker_pos, blob_det: callable = None, px_dist_thres: float = 25):
        self.blob_det: callable = get_default_tacthru_tracker() if blob_det is None else blob_det
        self.kpts_ref = ref_marker_pos

        self.kpts_ref_tree = KDTree(self.kpts_ref, metric="minkowski", p=2)
        self.kpts_buffer = np.zeros_like(self.kpts_ref)
        self.kpts_dist_buffer = np.zeros(len(self.kpts_ref))
        self.kpts_buffer_valid = np.zeros(len(self.kpts_ref), dtype=bool)
        self.px_dist_thres = px_dist_thres
        self.marker_id = np.arange(len(self.kpts_ref))

        self.reset()

    def reset(self):
        self.n_kpts = 0
        self.max_n_kpts = 64

        self.state_noise_cov = np.diag([0.01**2, 0.01**2])
        self.obs_noise_cov = np.diag([0.1**2, 0.1**2])
        self.kf_x = self.kpts_ref.copy()
        self.kf_A = self.kf_H = np.eye(2)
        self.kf_cov = np.zeros([len(self.kpts_ref), 2, 2], dtype=np.float32)

    def process(self, frame, ts: float = None):
        res = self.blob_det(frame)
        blobs = res.pop("blobs")
        res["all_kpts"] = kpts = np.asarray([kpt.pt for kpt in blobs])

        to_ref_dist, indices = self.kpts_ref_tree.query(kpts, k=1, return_distance=True)
        to_ref_dist, indices = to_ref_dist[:, 0], indices[:, 0]
        valid_n_kpt = len(np.unique(indices))

        to_kf_x_dist = np.linalg.norm(kpts - self.kf_x[indices], axis=1)
        sorted_indices = np.argsort(to_kf_x_dist)
        n_valid_pts = len(np.unique(indices))
        to_kf_x_dist = to_kf_x_dist[sorted_indices][:n_valid_pts]
        indices = indices[sorted_indices][:valid_n_kpt]
        kpts = kpts[sorted_indices][:valid_n_kpt]

        kf_pred = self.kf_x.copy()
        kf_obs = self.kf_x.copy()
        kf_obs[indices] = kpts

        kf_pred_cov = (
            self.kf_cov + self.state_noise_cov[None]
        )  # self.kf_A @ self.kf_cov[indices] @ self.kf_A.T + self.state_noise_cov
        kf_gain = kf_pred_cov @ np.linalg.inv(kf_pred_cov + self.obs_noise_cov[None])
        kf_update = kf_pred + (kf_gain @ (kf_obs - kf_pred)[..., None]).squeeze(-1)
        kf_update_cov = (np.eye(2)[None] - kf_gain) @ kf_pred_cov
        self.kf_x = kf_update
        self.kf_cov += self.state_noise_cov[None]
        self.kf_cov[indices] = kf_update_cov[indices]

        res.update({"marker": self.kf_x.copy(), "marker_ref": self.kpts_ref, "det_valid": valid_n_kpt})

        return res
