import sys
import os
from argparse import ArgumentParser
import numpy as np
from Bytetrack.yolox.tracker.byte_tracker import BYTETracker
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../..", "Botsort"))
from Botsort import BoTSORT


class Tracker(object):
    def __init__(self, model_name:str = "ByteTracker"):
        self.model_name = model_name
        if self.model_name == "ByteTracker":
            self.tracker_args = self.set_bytetracker_parser()
            self.tracker = BYTETracker(self.tracker_args, frame_rate=30.0)
        else:
            self.tracker_args = self.set_botsort_parser()
            self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)



    def process_bbox(self, image: np.ndarray,bboxes):
        # 將新偵測的邊界框更新到跟蹤器
        if self.model_name == "ByteTracker":
            online_targets = self.tracker.update(np.hstack((bboxes, np.full((bboxes.shape[0], 2), [0.9, 0]))), [image.shape[0], image.shape[1]], [image.shape[0], image.shape[1]])
            return online_targets

        online_targets = self.tracker.update(np.hstack((bboxes, np.full((bboxes.shape[0], 2), [0.9, 0]))), image)
        return online_targets

    def set_bytetracker_parser(self) -> ArgumentParser:
        parser = ArgumentParser()

        # tracking args
        parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
        parser.add_argument(
            "--aspect_ratio_thresh", type=float, default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value."
        )
        parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
        parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
        parser = parser.parse_args()
        return parser

    def set_botsort_parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        # tracking args
        parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
        parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
        parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
        parser.add_argument("--track_buffer", type=int, default=360, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
        parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                            help="threshold for filtering out boxes of which aspect ratio are above the given value.")
        parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
        parser.add_argument("--fuse-score", dest="mot20", default=True, action='store_true',
                            help="fuse score and iou for association")

        # CMC
        parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

        # ReID
        parser.add_argument("--with-reid", dest="with_reid", default=False , help="with ReID module.")
        parser.add_argument("--fast-reid-config", dest="fast_reid_config", default='../tracker/fast_reid/configs/MOT17/sbs_S50.yml',
                            type=str, help="reid config file path")
        parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default='../tracker/fast_reid/mot17_sbs_S50.pth',
                            type=str, help="reid config file path")
        parser.add_argument('--proximity_thresh', type=float, default=0.5,
                            help='threshold for rejecting low overlap reid matches')
        parser.add_argument('--appearance_thresh', type=float, default=0.25,
                            help='threshold for rejecting low appearance similarity reid matches')

        tracker_args = parser.parse_args()

        tracker_args.jde = False
        tracker_args.ablation = False
        return tracker_args