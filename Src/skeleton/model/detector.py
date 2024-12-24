from argparse import ArgumentParser
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# sys.path.append(os.path.join(current_dir, "../..", "tracker"))
from mmcv.transforms import Compose
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class Detector(object):
    def __init__(self):
        self.detect_args = self.setDetectParser()
        self.detector = init_detector(
        self.detect_args.det_config, self.detect_args.det_checkpoint, device=self.detect_args.device)
        self.detector.cfg.test_dataloader.dataset.pipeline[
            0].type = 'mmdet.LoadImageFromNDArray'
        self.detector_test_pipeline = Compose(self.detector.cfg.test_dataloader.dataset.pipeline)

    def setDetectParser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--det-config', default='./mmyolo_main/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py', help='Config file for detection')
        parser.add_argument('--det-checkpoint', default='../Db/pretrain/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth', help='Checkpoint file for detection')
        parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
        parser.add_argument(
            '--score-thr', type=float, default=0.3, help='Bbox score threshold')
        parser.add_argument(
            '--nms-thr',
            type=float,
            default=0.3,
            help='IoU threshold for bounding box NMS')
        parser.add_argument(
            '--kpt-thr',
            type=float,
            default=0.3,
            help='Visualizing keypoint thresholds')
        args = parser.parse_args()
        return args
    
