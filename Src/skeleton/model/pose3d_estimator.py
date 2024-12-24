from argparse import ArgumentParser
from mmpose.apis import init_model

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
            default='../Db/pretrain/motionbert_ft_h36m-d80af323_20230531.pth',
            help='Checkpoint file for the 2nd stage pose lifter model')
        parser.add_argument(
            '--disable-rebase-keypoint',
            action='store_true',
            default=False,
            help='Whether to disable rebasing the predicted 3D pose so its '
            'lowest keypoint has a height of 0 (landing on the ground). Rebase '
            'is useful for visualization when the model do not predict the '
            'global position of the 3D pose.')
        parser.add_argument(
            '--disable-norm-pose-2d',
            action='store_true',
            default=False,
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