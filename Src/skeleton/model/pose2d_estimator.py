from argparse import ArgumentParser
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
import numpy as np


class Pose2DEstimator(object):
    def __init__(self):
        self.pose2d_args = self.setPose2DParser()
        self.pose2d_estimator = init_model(
            self.pose2d_args.pose_config,
            self.pose2d_args.pose_checkpoint
        )

    def process_image(self, image:np.ndarray, bbox:np.array):
        """_summary_

        Args:
            image (np.ndarray): _description_
            bbox (np.array): _description_

        Returns:
            _type_: _description_
        """        

        pose_results = inference_topdown(self.pose2d_estimator, image, bbox)
        data_samples = merge_data_samples(pose_results)
        return pose_results, data_samples.get('pred_instances', None)

    def setPose2DParser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--pose-config', default='./mmpose_main/configs/body_2d_keypoint/topdown_heatmap/haple/ViTPose_base_simple_halpe_256x192.py', help='Config file for pose')
        parser.add_argument('--pose-checkpoint', default='../Db/pretrain/epoch_240.pth', help='Checkpoint file for pose')
        parser.add_argument(
            '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
            '--kpt-thr',
            type=float,
            default=0.3,
            help='Visualizing keypoint thresholds')
        args = parser.parse_args()
        return args