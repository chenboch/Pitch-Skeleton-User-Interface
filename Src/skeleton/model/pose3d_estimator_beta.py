from argparse import ArgumentParser
import numpy as np
from mmpose.apis import (init_model, convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown)
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)


class Pose3DEstimator(object):
    def __init__(self):
        self.pose3d_args = self.setPose3DParser()
        self.pose3d_estimator = init_model(
            self.pose3d_args.pose_lifter_config,
            self.pose3d_args.pose_lifter_checkpoint,
            device=self.pose3d_args.device.lower())

    def setPose3DParser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument(
            '--pose_lifter_config',
            default='./mmpose_main/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py',
            help='Config file for the 2nd stage pose lifter model')
        parser.add_argument(
            '--pose_lifter_checkpoint',
            default='../Db/checkpoints/motionbert_ft_h36m-d80af323_20230531.pth',
            help='Checkpoint file for the 2nd stage pose lifter model')
        parser.add_argument(
            '--disable-rebase-keypoint',
            action='store_true',
            default=True,
            help='Whether to disable rebasing the predicted 3D pose so its '
            'lowest keypoint has a height of 0 (landing on the ground). Rebase '
            'is useful for visualization when the model do not predict the '
            'global position of the 3D pose.')
        parser.add_argument(
            '--disable-norm-pose-2d',
            action='store_true',
            default=True,
            help='Whether to scale the bbox (along with the 2D pose) to the '
            'average bbox scale of the dataset, and move the bbox (along with the '
            '2D pose) to the average bbox center of the dataset. This is useful '
            'when bbox is small, especially in multi-person scenarios.')
        parser.add_argument(
            '--num-instances',
            type=int,
            default=1,
            help='The number of 3D poses to be visualized in every frame. If '
            'less than 0, it will be set to the number of pose results in the '
            'first frame.')
        parser.add_argument(
            '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
            '--online',
            action='store_true',
            default=False,
            help='Inference mode. If set to True, can not use future frame'
            'information when using multi frames for inference in the 2D pose'
            'detection stage. Default: False.')
        args = parser.parse_args()
        return args

    def process_pose3d(self ,pose_results, track_ids, img_shape):
        """
        將 2D 骨架關鍵點轉換為 3D 骨架關鍵點。

        Args:
            img_shape: 輸入影像尺寸。
            data_sample: 包含關鍵點的 2D 數據樣本。

        Returns:
            pred_3d_data_samples: 3D 預測骨架數據樣本。
        """
        # 提取數據集名稱
        pose_det_dataset_name = "halpe26"
        pose_lift_dataset_name = self.pose3d_estimator.dataset_meta['dataset_name']
        pose_lift_dataset = self.pose3d_estimator.cfg.test_dataloader.dataset
        # 初始化 2D 骨架轉換的結果容器
        pose_est_results_list = []
        pose_est_results_converted = []

        for i, data_sample in enumerate(pose_results):
            pred_instances = data_sample.cpu().numpy()
            keypoints = pred_instances.keypoints
            pose_results[i].set_field(track_ids[i], 'track_id')
            # 步驟 1: 轉換關鍵點格式
            pose_est_result_converted = self.convert_keypoints(
                pose_results[i], keypoints, pose_det_dataset_name, pose_lift_dataset_name
            )
            pose_est_results_converted.append(pose_est_result_converted)
        pose_est_results_list.append(pose_est_results_converted)
        # 步驟 2: 提取 2D 骨架序列
        pose_seq_2d = extract_pose_sequence(
                            pose_est_results_list,
                            frame_idx=0,
                            causal=pose_lift_dataset.get('causal', False),
                            seq_len=pose_lift_dataset.get('seq_len', 1),
                            step=pose_lift_dataset.get('seq_step', 1)
                        )
        # 步驟 3: 進行 2D-to-3D 提升

        pose_lift_results = self._lift_to_3d(pose_seq_2d, img_shape[:2])
        # 步驟 4: 後處理 3D 骨架數據
        pose_lift_results = self.postprocess_pose_lift(pose_lift_results, pose_results)
        # 合併樣本並返回結果
        pred_3d_data_samples = merge_data_samples(pose_lift_results)
        return pred_3d_data_samples.get('pred_instances', None)

    def convert_keypoints(self, pose_result, keypoints, det_name, lift_name):
        """轉換 2D 關鍵點的數據格式。"""
        converted_sample = PoseDataSample()
        converted_sample.set_field(pose_result.clone(), 'pred_instances')
        converted_sample.set_field(pose_result.clone(), 'gt_instances')

        # 轉換關鍵點定義
        keypoints = convert_keypoint_definition(keypoints, det_name, lift_name)
        converted_sample.pred_instances.set_field(keypoints, 'keypoints')
        converted_sample.set_field(pose_result.track_id, 'track_id')
        return converted_sample

    def _lift_to_3d(self, pose_seq_2d, image_size):
        """使用模型進行 2D-to-3D 提升。"""
        norm_pose_2d = not self.pose3d_args.disable_norm_pose_2d
        return inference_pose_lifter_model(
            self.pose3d_estimator,
            pose_seq_2d,
            image_size=image_size,
            norm_pose_2d=norm_pose_2d
        )

    def postprocess_pose_lift(self, pose_lift_results, pose_results):
        """後處理提升的 3D 骨架數據。"""
        for idx, pose_lift_result in enumerate(pose_lift_results):
            pose_lift_result.track_id = pose_results[idx].get('track_id', 1e4)

            pred_instances = pose_lift_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_lift_results[idx].pred_instances.keypoint_scores = keypoint_scores
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]

            # rebase height (z-axis)
            # if not args.disable_rebase_keypoint:
            keypoints[..., 2] -= np.min(
                keypoints[..., 2], axis=-1, keepdims=True)

            pose_lift_results[idx].pred_instances.keypoints = keypoints
        pose_lift_results = sorted(
                pose_lift_results, key=lambda x: x.get('track_id', 1e4))

        return pose_lift_results