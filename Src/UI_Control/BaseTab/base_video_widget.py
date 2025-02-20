from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QObject, QTimer
from typing import Optional
import numpy as np
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

class BasePoseVideoTab(QWidget, AbstractPoseBase):
    def __init__(self, wrapper:Wrapper, model_name: str, parent: Optional[QWidget] = None):
        super(BasePoseVideoTab, self).__init__(parent)
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

    def resize_event(self, event):
        new_size = event.size()
        # 在此執行你想要的操作
        if self.video_loader.video_name is not None:
            self.update_frame(self.ui.frame_slider.value())
        super().resize_event(event)

    def init_frame_slider(self):
        """初始化影片滑桿和相關的標籤。"""
        total_frames = self.video_loader.total_frames
        self.ui.frame_slider.setMinimum(0)
        self.ui.frame_slider.setMaximum(total_frames - 1)
        self.ui.frame_slider.setValue(0)
        self.ui.frame_num_label.setText(f'0/{total_frames - 1}')

    def init_graph(self):
        """初始化圖表和模型設定。"""
        total_frames = self.video_loader.total_frames
        self.graph_plotter._init_graph(total_frames)
        self.show_graph(self.curve_scene, self.ui.curve_view)

    def play_btn_clicked(self):
        if self.video_loader.video_name == "":
            QMessageBox.warning(self, "無法播放影片", "請讀取影片!")
            return
        if self.video_loader.is_loading:
            QMessageBox.warning(self, "影片讀取中", "請稍等!")
            return
        self.is_play = not self.is_play
        self.ui.play_btn.setText("||" if self.is_play else "▶︎")
        if self.is_play:
            self.play_frame(self.ui.frame_slider.value())

    def mouse_press_event(self, event):
        view_rect = self.ui.frame_view.rect()
        pos = event.pos()

        if not view_rect.contains(pos):
            return
        search_person_df = self.pose_estimater.get_person_df(frame_num = self.ui.frame_slider.value())
        scene_pos = self.ui.frame_view.mapToScene(pos)
        x, y = scene_pos.x(), scene_pos.y()

        if self.ui.select_checkbox.isChecked() and not self.kpt_table.label_kpt :
            if event.button() == Qt.LeftButton:
                try:
                    self.person_selector.select(search_person_df, x, y)
                    self.pose_estimater.track_id = self.person_selector.selected_id
                except IndexError:
                    self.logger.info("請重新選人")
                    self.ui.select_checkbox.setCheckState(0)

        if self.ui.select_kpt_checkbox.isChecked() and not self.kpt_table.label_kpt :
            if event.button() == Qt.LeftButton:
                self.kpt_selector.select(search_person_df, x, y)
                self.pose_estimater.joint_id = self.kpt_selector.selected_id

        if self.kpt_table.label_kpt:
            if event.button() == Qt.LeftButton:
                self.kpt_table.sendToTable(x, y, 1, self.ui.frame_slider.value())
            elif event.button() == Qt.RightButton:
                self.kpt_table.sendToTable(0, 0, 0, self.ui.frame_slider.value())
        if self.video_loader.video_name is not None:
            self.update_frame(frame_num=self.ui.frame_slider.value())

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

    def load_video(self,is_processed:bool = False):
        if self.is_play:
            self.ui.play_btn.click()
        self.is_processed = is_processed
        self.video_loader.load_video()
        self.check_video_load()

    def load_processed_data(self):
        json_loader = JsonLoader(self.video_loader.folder_path, self.video_loader.video_name, self.model_name)
        json_loader.load_json()
        self.pose_estimater.person_df = json_loader.person_df
        self.ui.show_skeleton_checkbox.setChecked(True)

    def check_video_load(self):
        """檢查影片是否讀取完成，並更新 UI 元素。"""
        # 檢查是否有影片名稱，若無則不執行後續操作
        if not self.video_loader.video_name:
            return
        # 若影片正在讀取中，定時檢查讀取狀況
        if self.video_loader.is_loading:
            self.ui.video_name_label.setText("讀取影片中")
            QTimer.singleShot(100, self.check_video_load)  # 每100ms 檢查一次
            return
        # 影片讀取完成後更新 UI 元素
        self.update_video_info()

    def update_video_info(self):
        """更新與影片相關的資訊顯示在 UI 上。"""
        self.reset()
        self.turn_off_all_checkbox()
        self.init_frame_slider()
        self.init_graph()
        self.update_frame(0)
        self.ui.video_name_label.setText(self.video_loader.video_name)
        video_size = self.video_loader.video_size
        self.ui.resolution_label.setText(f"(0,0) - {video_size[0]} x {video_size[1]}")
        if self.is_processed:
            self.load_processed_data()

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

    def play_frame(self, start_num:int=0):
        for i in range(start_num, self.video_loader.total_frames):
            if not self.is_play:
                break
            self.ui.frame_slider.setValue(i)
            if i == self.video_loader.total_frames - 1 and self.is_play:
                self.play_btn_clicked()
            cv2.waitKey(15)

    def analyze_frame(self):
        fps = 0
        frame_num = self.ui.frame_slider.value()
        self.ui.frame_num_label.setText(f'{frame_num}/{len(self.video_loader.video_frames) - 1}')
        frame = self.video_loader.get_video_image(frame_num)
        fps= self.pose_estimater.detect_keypoints(frame, frame_num)
        self.ui.fps_info_label.setText(f"{fps:02d}")

        if self.pose_estimater.track_id is not None:
            self.pose_analyzer.addAnalyzeInfo(frame_num)
            self.graph_plotter.updateGraph(frame_num)
            self.kpt_table.importDataToTable(frame_num)
        if frame_num == self.video_loader.total_frames - 1:
            self.video_loader.save_video(self.model_name)
        self.update_frame(frame_num)

    def update_frame(self, frame_num:int):
        image = self.video_loader.get_video_image(frame_num)
        drawed_img = self.image_drawer.drawInfo(image, frame_num, self.pose_estimater.kpt_buffer)
        self.show_image(drawed_img, self.view_scene, self.ui.frame_view)
        self.graph_plotter.resize_graph(self.ui.curve_view.width(),self.ui.curve_view.height())


    def toggle_detect(self):
        self.ui.show_skeleton_checkbox.setChecked(True)
        frame = self.video_loader.get_video_image(0)
        fps = self.pose_estimater.detect_keypoints(frame, 0)
        self.ui.play_btn.click()

    def toggle_select(self, state:int):
        if not self.ui.show_skeleton_checkbox.isChecked():
            self.ui.select_checkbox.setCheckState(0)
            self.logger.warning("無法選擇人，請選擇顯示人體骨架!")
            return
        if state == 0:
            self.pose_estimater.track_id = None
            self.ui.select_kpt_checkbox.setCheckState(0)
        self.update_frame(self.ui.frame_slider.value())

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
        self.update_frame(self.ui.frame_slider.value())

    def toggle_show_skeleton(self, state:int):
        is_checked = state == 2
        self.pose_estimater.is_detect = is_checked
        self.image_drawer.set_show_skeleton(is_checked)
        if state == 0:
            self.ui.select_checkbox.setCheckState(0)
        self.update_frame(self.ui.frame_slider.value())

    def toggle_show_bbox(self, state:int):
        if state == 2:
            self.image_drawer.show_bbox = True
        else:
            self.image_drawer.show_bbox = False

    def toggle_show_angle_info(self, state:int):
        if not self.ui.select_checkbox.isChecked():
            self.ui.show_angle_checkbox.setCheckState(0)
            QMessageBox.warning(self, "無法顯示關節點角度資訊", "請選擇人!")
            return
        if state == 2:
            self.image_drawer.set_show_angle_info(True)
        else:
            self.image_drawer.set_show_angle_info(False)

    def correctId(self):
        before_correctId = self.ui.beforeCorrectId.value()
        after_correctId = self.ui.afterCorrectId.value()
        self.pose_estimater.correct_person_id(before_correctId, after_correctId)
        self.update_frame(self.ui.frame_slider.value())

    def turn_off_all_checkbox(self):
        # self.ui.select_kpt_checkbox.setCheckState(0)
        # self.ui.select_checkbox.setCheckState(0)
        self.ui.show_skeleton_checkbox.setCheckState(0)
        # self.ui.show_bbox_checkbox.setCheckState(0)
        # self.ui.show_angle_checkbox.setCheckState(0)