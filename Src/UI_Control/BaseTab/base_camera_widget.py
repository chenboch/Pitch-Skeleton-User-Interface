from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QObject, QTimer
from typing import Optional
import numpy as np
import os
from datetime import datetime
import cv2
from ..utils import *
from ..cv_utils import *
from skeleton import Wrapper
from ..vis_utils import *

import pyqtgraph as pg
from abc import ABC ,ABCMeta, abstractmethod
import logging
from sip import wrapper

class SipABCMeta(ABCMeta, type(wrapper)):
    """結合 ABCMeta 和 sip.wrapper 的 metaclass。"""
    pass

# 定義抽象功能類
class AbstractPoseBase(QObject, metaclass=SipABCMeta):
    @abstractmethod
    def setup_pose_estimater(self):
        pass

class BasePoseCameraTab(QWidget, AbstractPoseBase):
    def __init__(self, wrapper:Wrapper, model_name: str, parent: Optional[QWidget] = None):
        super(BasePoseCameraTab, self).__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)  # 獲取當前類的日誌對象
        # self.logger.info("PoseEstimater initialized with wrapper.")
        self.ui = None
        self.wrapper = wrapper
        self.model_name = model_name
        self.is_processed = False
        self.view_scene = QGraphicsScene()
        self.curve_scene = QGraphicsScene()
        self.setup_components()
        self.init_var()

    def init_var(self):
        self.is_play = False
        self.view_scene.clear()
        self.curve_scene.clear()
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.camera = Camera()
        self.timer = QTimer()
        self.timer.timeout.connect(self.analyze_frame)

    # def resize_event(self, event):
    #     new_size = event.size()
    #     # 在此執行你想要的操作
    #     if self.video_loader.video_name is not None:
    #         self.update_frame(self.ui.frame_slider.value())
    #     super().resize_event(event)

    def change_camera(self):
        """Change the camera based on input value."""
        self.camera.set_camera_id(self.ui.camera_id_input.value())

    def toggle_camera(self, state:int):
        """Toggle the camera on/off based on checkbox state."""
        if state == 2:
            frame_width, frame_height, fps = self.camera.toggle_camera(True)
            self.ui.resolution_label.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
            self.timer.start(1)
        else:
            self.camera.toggle_camera(False)
            self.timer.stop()

    def toggle_record(self, state:int):
        """Start or stop video recording."""
        if state == 2:
            self.start_recording()
            self.ui.show_skeleton_checkbox.setChecked(False)
        else:
            self.camera.stop_recording()

    def start_recording(self):
        """Start recording the video."""
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f'../../Db/Record/C{self.ui.camera_id.value()}_Fps120_{current_time}'
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.join(output_dir, f'C{self.ui.camera_id.value()}_Fps120_{current_time}.mp4')
        self.ui.show_skeleton_checkbox.setChecked(False)
        self.camera.start_recording(video_filename)

    # def init_graph(self):
    #     """初始化圖表和模型設定。"""
    #     total_frames = self.video_loader.total_frames
    #     self.graph_plotter._init_graph(total_frames)
    #     self.show_graph(self.curve_scene, self.ui.curve_view)

    def mouse_press_event(self, event):
        view_rect = self.ui.frame_view.rect()
        pos = event.pos()

        if not view_rect.contains(pos):
            return

        search_person_df = self.pose_estimater.get_person_df()
        scene_pos = self.ui.frame_view.mapToScene(pos)
        x, y = scene_pos.x(), scene_pos.y()

        if self.ui.select_checkbox.isChecked():
            if event.button() == Qt.LeftButton:
                try:
                    self.person_selector.select(search_person_df, x, y)
                    self.pose_estimater.track_id = self.person_selector.selected_id
                except IndexError:
                    self.logger.info("請重新選人")
                    self.ui.select_checkbox.setCheckState(0)

        if self.ui.select_kpt_checkbox.isChecked():
            if event.button() == Qt.LeftButton:
                self.kpt_selector.select(search_person_df, x, y)
                self.pose_estimater.joint_id = self.kpt_selector.selected_id

    def setup_components(self):
        self.setup_pose_estimater()  # 由子類別提供具體的 pose_estimater
        self.person_selector = PersonSelector()
        self.kpt_selector = KptSelector()
        self.pose_analyzer = PoseAnalyzer(self.pose_estimater)
        self.graph_plotter = GraphPlotter(self.pose_analyzer)
        self.image_drawer = ImageDrawer(self.pose_estimater, self.pose_analyzer)
        self.video_loader = VideoLoader(self.image_drawer)

    def reset(self):
        self.person_selector.reset()
        self.kpt_selector.reset()
        self.pose_estimater.reset()
        self.pose_analyzer.reset()
        self.graph_plotter.reset()
        self.image_drawer.reset()
        self.view_scene.clear()
        self.curve_scene.clear()

    def show_image(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        scene.clear()
        image = image.copy()
        image = cv2.circle(image, (0, 0), 10, (0, 0, 255), -1)
        w, h = image.shape[1], image.shape[0]
        bytesPerline = 3 * w
        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def show_graph(self, scene:QGraphicsScene, graphicview:QGraphicsView):
        scene.clear()
        graph = self.graph_plotter.graph
        graph.resize(graphicview.width(),graphicview.height())
        scene.addWidget(graph)
        graphicview.setScene(scene)
        graphicview.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def analyze_frame(self):
        """Analyze and process each frame from the camera."""
        if len(self.camera.frame_buffer) > 0:
            frames = list(self.camera.frame_buffer)
            if len(frames) < 5:
                last_frame = frames[-1] if frames else None
                if last_frame is None:
                    self.logger.error("緩衝區無幀，無法填充")
                    return None
                # 填充至 5 幀
                frames = frames + [last_frame] * (5 - len(frames))
                self.logger.debug(f"已用最後一幀填充至 5 幀")
            # print(len(frames))
            fps = self.pose_estimater.detect_keypoints(np.array(frames))
            self.ui.fps_info_label.setText(f"{fps:02d}")
            self.update_frame(frames[2])

    def update_frame(self, frame:np.array):
        """Update the displayed frame with additional analysis."""
        drawed_img = self.image_drawer.drawInfo(img = frame, kpt_buffer = self.pose_estimater.kpt_buffer, is_realtime= True)
        self.show_image(drawed_img, self.view_scene, self.ui.frame_view)

    def toggle_select(self, state:int):
        if not self.ui.show_skeleton_checkbox.isChecked():
            self.ui.select_checkbox.setCheckState(0)
            self.logger.warning("無法選擇人，請選擇顯示人體骨架!")
            return
        if state == 0:
            self.pose_estimater.track_id = None
            self.ui.select_kpt_checkbox.setCheckState(0)

    def toggle_kpt_select(self, state:int):
        """Toggle keypoint selection and trajectory visualization."""
        if not self.ui.select_checkbox.isChecked():
            self.ui.select_kpt_checkbox.setCheckState(0)
            self.logger.warning("無法選擇關節點， 請先選擇人!")
            return
        if state == 2:
            self.pose_estimater.joint_id = 10
            self.image_drawer.set_show_traj(True)
        else:
            self.pose_estimater.joint_id = None
            self.image_drawer.set_show_traj(False)

    def toggle_show_skeleton(self, state:int):
        is_checked = state == 2
        self.pose_estimater.is_detect = is_checked
        self.image_drawer.set_show_skeleton(is_checked)
        # self.camera.set_fps_control(10 if is_checked else 1)
        if state == 0:
            self.ui.select_checkbox.setCheckState(0)

    def toggle_show_bbox(self, state:int):
        if state == 2:
            self.image_drawer.show_bbox = True
        else:
            self.image_drawer.show_bbox = False

    def toggle_showgrid(self, state:int):
        """Toggle gridline visibility."""
        if state == 2:
            self.image_drawer.show_grid = True
        else:
            self.image_drawer.show_grid = False

    def toggle_show_angle_info(self, state:int):
        if not self.ui.select_checkbox.isChecked():
            self.ui.show_angle_checkbox.setCheckState(0)
            QMessageBox.warning(self, "無法顯示關節點角度資訊", "請選擇人!")
            return
        if state == 2:
            self.image_drawer.set_show_angle_info(True)
        else:
            self.image_drawer.set_show_angle_info(False)

    def turn_off_all_checkbox(self):
        self.ui.show_skeleton_checkbox.setCheckState(0)