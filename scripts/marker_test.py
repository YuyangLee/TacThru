import sys
import time

import cv2
import numpy as np

sys.path.append(".")

from utils.proc_utils import KeypointsKFProcessor

tacthru_ref_marker_pos = np.load("data/tacthru/ref_kpts.npy") * 400  # 400 is the image resolution
tracker = KeypointsKFProcessor(tacthru_ref_marker_pos)

data_files = [
    "data/marker_tests/test-1/GelSight-0.avi",
    "data/marker_tests/test-2/GelSight-0.avi",
]

for video_path in data_files:
    print(f"Processing {video_path}")
    video = cv2.VideoCapture(video_path)

    tracker.reset()

    i_frame = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_viz = frame.copy()

        time_start = time.time()
        res_dict = tracker(frame)
        proc_time = time.time() - time_start

        all_kpts = res_dict["all_kpts"]
        kpts, kpts_ref = res_dict["marker"], res_dict.get("marker_ref")
        n_markers = len(kpts)
        kpts_dist = np.linalg.norm(kpts - kpts_ref, axis=-1)

        thres = np.asarray(res_dict["thres"])[..., None].repeat(3, axis=-1)
        thres_det_all = thres.copy()

        for kpt in all_kpts:
            kpt = kpt.astype(int)
            cv2.circle(thres_det_all, kpt, 5, (0, 0, 255), 1, 1)
        cv2.putText(
            thres_det_all,
            f"{len(all_kpts)} keypoints detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        for i, (kpt, kpt_ref) in enumerate(zip(kpts.astype(int), kpts_ref.astype(int))):
            cv2.arrowedLine(frame_viz, kpt_ref, ((kpt - kpt_ref) * 4 + kpt_ref), (255, 255, 0), 3, tipLength=0.5)

        frame_output = np.hstack([frame, thres, thres_det_all, frame_viz])

        cv2.imshow("Detected markers", frame_output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
