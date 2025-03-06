from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QObject, QTimer
from typing import Optional
import numpy as np
import cv2
from UI_Control.utils import *
from UI_Control.cv_utils import *
from skeleton import Wrapper
from UI_Control.vis_utils import *
from .video_ui import Ui_video_widget

class PoseVideoTabControl(QWidget):
    def __init__(self, parent = None):
        super(PoseVideoTabControl, self).__init__(parent)
        self.ui = Ui_video_widget()

        self.ui.setupUi(self)
        self.left_view_scene = QGraphicsScene()
        self.right_view_scene = QGraphicsScene()
        self.setup_components()
        self.init_var()
        self.bind_ui()

    def init_var(self):
        self.is_play = False
        self.left_view_scene.clear()
        self.right_view_scene.clear()


    def bind_ui(self):
        self.ui.load_original_video_btn.clicked.connect(
            lambda: self.load_video(is_left=True))
        self.ui.load_original_video_btn_2.clicked.connect(
            lambda: self.load_video(is_left=False))
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.update_frame)

    def setup_components(self):
        self.left_video_loader = VideoLoader()
        self.right_video_loader = VideoLoader()

    def update_frame(self, frame_num:int):
        self.ui.frame_num_label.setText(f'{frame_num}/{len(self.left_video_loader.video_frames) - 1}')
        if self.left_video_loader.video_name:
            image = self.left_video_loader.get_video_image(frame_num)
            self.show_image(image, self.left_view_scene, self.ui.left_frame_view)
        if self.right_video_loader.video_name:
            image = self.right_video_loader.get_video_image(frame_num)
            self.show_image(image, self.right_view_scene, self.ui.right_frame_view)

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

    def load_video(self,is_left:bool = False):
        if self.is_play:
            self.ui.play_btn.click()
        # self.is_processed = is_left
        video_loader = self.left_video_loader if is_left else self.right_video_loader
        video_loader.load_video()
        self.check_video_load(video_loader, is_left)

    def check_video_load(self, video_loader, is_left:bool=False):
        """檢查影片是否讀取完成，並更新 UI 元素。"""
        # 檢查是否有影片名稱，若無則不執行後續操作
        if not video_loader.video_name:
            return
        # 若影片正在讀取中，定時檢查讀取狀況
        if video_loader.is_loading:
            self.ui.video_name_label.setText("讀取影片中")
            QTimer.singleShot(100, lambda: self.check_video_load(video_loader, is_left))
            return
        # 影片讀取完成後更新 UI 元素
        self.update_video_info(video_loader, is_left)

    def play_btn_clicked(self):
        video_loader = self.left_video_loader  # 或根據 UI 狀態選擇
        if video_loader.video_name == "":
            QMessageBox.warning(self, "無法播放影片", "請讀取影片!")
            return
        if video_loader.is_loading:
            QMessageBox.warning(self, "影片讀取中", "請稍等!")
            return
        self.is_play = not self.is_play
        self.ui.play_btn.setText("||" if self.is_play else "▶︎")
        if self.is_play:
            self.play_frame(self.ui.frame_slider.value())

    def play_frame(self, start_num:int=0):
        for i in range(start_num, self.left_video_loader.total_frames):
            if not self.is_play:
                break
            self.ui.frame_slider.setValue(i)
            if i == self.left_video_loader.total_frames - 1 and self.is_play:
                self.play_btn_clicked()
            cv2.waitKey(15)

    def update_video_info(self, video_loader, is_left:bool = False):
        """更新與影片相關的資訊顯示在 UI 上。"""
        self.init_frame_slider(video_loader)
        self.update_frame(0)
        if is_left:
            self.ui.video_name_label.setText(video_loader.video_name)
        else:
            self.ui.video_name_label_2.setText(video_loader.video_name)
        video_size = video_loader.video_size
        self.ui.resolution_label.setText(f"(0,0) - {video_size[0]} x {video_size[1]}")

    def init_frame_slider(self, video_loader):
        """初始化影片滑桿和相關的標籤。"""
        total_frames = video_loader.total_frames
        self.ui.frame_slider.setMinimum(0)
        self.ui.frame_slider.setMaximum(total_frames - 1)
        self.ui.frame_slider.setValue(0)
        self.ui.frame_num_label.setText(f'0/{total_frames - 1}')

    def keyPressEvent(self, event):
        key = event.text().lower()  # 用 event.text() 抓字元
        if key == 'd':
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        elif key == 'a':
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        else:
            super().keyPressEvent(event)


