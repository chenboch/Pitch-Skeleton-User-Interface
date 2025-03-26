# Copyright (c) OpenMMLab. All rights reserved.

# 導入子模組功能
# from .cv_utils.cv_thread import (
#     VideoCaptureThread, VideoWriterThread, VideoToImagesThread
# )
# from .cv_utils.cv_control import Camera
from .Main_UI import *
# from .Pose2DTab.Camera_Widget.camera_widget import (
#     PoseCameraTabControl as Pose2DCameraTabControl
# )
from .Pose2DTab.Video_Widget.video_widget import (
    PoseVideoTabControl as Pose2DVideoTabControl
)

from .Pose2DTab.Pitch_Widget.video_widget import (
    PoseVideoTabControl as Pose2DSyncVideoTabControl
)
# from .Pose2DTab.Pitch_Widget.pitch_widget import (
#     PosePitchTabControl as Pose2DPitchTabControl
# )
# from .Pose3DTab.Video_Widget.video_widget import (
#     PoseVideoTabControl as Pose3DVideoTabControl
# )
from .vis_utils.vis_image import ImageDrawer
from .utils.timer import Timer
from .ui_utils import *

# 將模組按功能分類，提升可讀性
__all__ = [
    # CV 功能模組
    'cv_control', 'Camera', 'VideoCaptureThread', 'VideoWriterThread', 'VideoToImagesThread',

    # Pose2D 功能模組
    'Pose2DCameraTabControl', 'Pose2DVideoTabControl', 'Pose2DPitchTabControl',

    # Pose3D 功能模組
    'Pose3DVideoTabControl',

    # 工具模組
    'ImageDrawer', 'Timer', 'ui_utils',

    # 主 UI
    'Main_UI',
]
