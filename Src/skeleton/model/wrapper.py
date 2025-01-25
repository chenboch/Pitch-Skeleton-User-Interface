"""Wrapper 模組

這個模組提供一個 Wrapper 類別，將 Detector、Tracker、Pose2DEstimator
和 Pose3DEstimator 統一封裝在一個介面中。
"""

from .detector import Detector
from .tracker import Tracker
from .pose2d_estimator import Pose2DEstimator
from .pose3d_estimator import Pose3DEstimator


class Wrapper:
    """封裝檢測器、追蹤器、2D 和 3D 姿態估計器的類別。

    提供統一的介面來初始化和訪問這些元件。
    """

    def __init__(self):
        """初始化 Wrapper 類別，建立 Detector、Tracker 等實例。"""
        self._detector = Detector()
        self._tracker = Tracker()
        self._pose2d_estimator = Pose2DEstimator()
        self._pose3d_estimator = Pose3DEstimator()

    @property
    def detector(self):
        """返回檢測器實例。"""
        return self._detector

    @property
    def tracker(self):
        """返回追蹤器實例。"""
        return self._tracker

    @property
    def pose2d_estimator(self):
        """返回 2D 姿態估計器實例。"""
        return self._pose2d_estimator

    @property
    def pose3d_estimator(self):
        """返回 3D 姿態估計器實例。"""
        return self._pose3d_estimator
