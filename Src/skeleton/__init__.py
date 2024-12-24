# skeleton/__init__.py
from .model.wrapper import Wrapper
from .camera_demo.skeleton_estimator import PoseEstimater as CameraPose2DEstimater
from .video_demo.skeleton_estimator import PoseEstimater as VidePose2DEstimater
from .video_demo.skeleton_lifter import PoseLifter as VidePose3DEstimater
# from .datasets.

__all__ = ['Wrapper','CameraPose2DEstimater','VidePose2DEstimater', 'VidePose3DEstimater']
