from argparse import ArgumentParser, REMAINDER
from mmpose.apis import init_model
from mmpose.apis import inference_topdown as vitpose_inference_topdown
from mmpose.structures import (merge_data_samples)
from mmpose.structures.pose_data_sample import InstanceData
from FAMI_Pose.tools import init_pose_model
from FAMI_Pose.tools import inference_topdown as famipose_inference_topdown
from skeleton.utils import FPSTimer
import numpy as np
import torch
import os

class Pose2DEstimator(object):
    def __init__(self, model_name:str = "vit-pose"):
        self._model_name = model_name
        self.fps_timer = FPSTimer()
        self.pose2d_cfg = None
        if model_name == "vit-pose":
            self.pose2d_args = self.set_vitpose_parser()
            self.pose2d_estimator = init_model(
                self.pose2d_args.pose_config,
                self.pose2d_args.pose_checkpoint
            )
        else:
            self.pose2d_args = self.set_famipose_parser()
            self.pose2d_estimator, self.pose2d_cfg= init_pose_model(
                self.pose2d_args
            )
            model_name = self.pose2d_args.weight
            self._model_name = model_name.split('\\')[-1].replace('.pth', '')

    def process_image(self, image_array:np.ndarray, bbox:np.array) -> list:
        """_summary_

        Args:
            image (np.ndarray): _description_
            bbox (np.array): _description_

        Returns:
            _type_: _description_
        """
        if self._model_name == "vit-pose":
            image = image_array[2]
            # self.fps_timer.tic()
            pose_results = vitpose_inference_topdown(self.pose2d_estimator, image, bbox)
            # self.fps_timer.toc()
            # print(f"tracking time: {self.fps_timer.time_interval}, fps: {int(self.fps_timer.fps) if int(self.fps_timer.fps)  < 100 else 0}")

            data_samples = merge_data_samples(pose_results)
            return data_samples.get('pred_instances', None)
        image_size = np.array(self.pose2d_cfg.MODEL.IMAGE_SIZE)

        pose_results = famipose_inference_topdown(self.pose2d_estimator, image_array, np.array(bbox), image_size)
        # 使用列表保存預處理的結果
        bboxes = []
        keypoints = []
        keypoint_scores = []

        # 將每個 `pose_result` 預處理並存儲
        for _, pose_result in enumerate(pose_results):
            # 預先將結果存儲在列表中，避免多次的 torch.tensor 和 numpy 轉換
            bboxes.append(pose_result[0])
            keypoints.append(pose_result[1][..., :2])  # 只取前兩個元素，對應 (x, y)
            keypoint_scores.append(pose_result[1][..., 2])  # 只取分數

        # 一次性使用 torch.stack 將所有 bboxes、keypoints、scores 合併為 tensor
        pred_instances = InstanceData()
        pred_instances.bboxes = torch.from_numpy(np.array(bboxes))
        pred_instances.keypoints = torch.from_numpy(np.array(keypoints))
        pred_instances.keypoint_scores = torch.from_numpy(np.array(keypoint_scores))
        return pred_instances

    def set_vitpose_parser(self) -> ArgumentParser:
        """_summary_

        Returns:
            ArgumentParser: _description_
        """

        parser = ArgumentParser()
        parser.add_argument('--pose-config', default='./mmpose_main/configs/body_2d_keypoint/topdown_heatmap/haple/ViTPose_base_simple_halpe_256x192.py', help='Config file for pose')
        parser.add_argument('--pose-checkpoint', default='../Db/checkpoints/vitpose.pth', help='Checkpoint file for pose')
        parser.add_argument(
            '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
            '--kpt-thr',
            type=float,
            default=0.3,
            help='Visualizing keypoint thresholds')
        args = parser.parse_args()
        return args

    def set_famipose_parser(self) -> ArgumentParser:
        root_dir = os.path.abspath('../')
        parser = ArgumentParser(description='Inference pose estimation Network')
        parser.add_argument('--cfg', help='experiment configure file name', required=False, type=str,
                            # Src\fami_main\configs\posetimation\fami\posetrack17\model_inference_hrnet.yaml
                            default="Src/FAMI_Pose/configs/Alignment/posetrack17/Alignment_V24.yaml")
        parser.add_argument('--PE_Name', help='pose estimation model name', required=False, type=str,
                            default='fami')
        parser.add_argument('-weight', help='model weight file', required=False, type=str
                            , default='Db/checkpoints/0520_24_epoch_19_state.pth')
                            # , default='Db/checkpoints/0531_25_epoch_19_state.pth')

        parser.add_argument('--gpu_id', default='0')
        parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=REMAINDER)

        args = parser.parse_args()
        args.rootDir = root_dir
        args.cfg = os.path.abspath(os.path.join(args.rootDir, args.cfg))
        args.weight = os.path.abspath(os.path.join(args.rootDir, args.weight))
        return args

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name