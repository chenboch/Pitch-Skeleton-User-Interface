from vispy import scene
from UI_Control.BaseTab import BasePoseVideoTab
from UI_Control.ui_utils import KeypointTable
from skeleton import VidePose3DEstimater
from .video_ui import Ui_video_widget
from .canvas3d_widget import Canvas3DView
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QWidget, QGraphicsPixmapItem

import cv2
from PyQt5.QtGui import QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QObject, QTimer
import numpy as np

class PoseVideoTabControl(BasePoseVideoTab):
    def __init__(self, wrapper, parent = None):
        super().__init__(wrapper, parent)
        self.ui = Ui_video_widget()
        self.ui.setupUi(self)
        self.bind_ui()
        self.setup_canvas_view()



    def bind_ui(self):
        self.ui.curve_view.setVisible(False)
        self.ui.load_original_video_btn.clicked.connect(
            lambda: self.load_video(is_processed=False))
        self.ui.load_processed_video_btn.clicked.connect(
            lambda: self.load_video(is_processed=True))

        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame)
        self.kpt_table = KeypointTable(self.ui.kpt_table,self.pose_estimater)
        self.ui.kpt_table.cellActivated.connect(self.kpt_table.onCellClicked)
        self.ui.frame_view.mousePressEvent = self.mouse_press_event
        self.ui.id_correct_btn.clicked.connect(self.correctId)
        self.ui.start_code_btn.clicked.connect(self.toggle_detect)
        self.ui.select_checkbox.stateChanged.connect(self.toggle_select)
        self.ui.show_skeleton_checkbox.stateChanged.connect(self.toggle_show_skeleton)
        self.ui.select_kpt_checkbox.stateChanged.connect(self.toggle_kpt_select)
        self.ui.show_bbox_checkbox.stateChanged.connect(self.toggle_show_bbox)
        self.ui.show_angle_checkbox.stateChanged.connect(self.toggle_show_angle_info)

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


            curr_df = self.pose_estimater.get_person_df(frame_num, is_select=True, is_kpt3d=True)
            if curr_df is not None:
                self.ui.canvas_3d_view.update_points(pos=curr_df)  # 更新 3D 點位

        if frame_num == self.video_loader.total_frames - 1:
            self.video_loader.save_video(self.model_name)
        self.update_frame(frame_num)

    def setup_pose_estimater(self):
        """Setup 2D pose estimator."""
        self.pose_estimater = VidePose3DEstimater(self.wrapper)


    def show_image(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        scene.clear()
        image = image.copy()
        image = cv2.circle(image, (0, 0), 10, (0, 0, 255), -1)

        # 轉換圖像
        h, w = image.shape[:2]
        bytesPerline = 3 * w
        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)

        # 設置 Scene 給 GraphicsView
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

        # **確保 GraphicsView 完成更新**
        GraphicsView.viewport().update()
        GraphicsView.updateGeometry()
        GraphicsView.repaint()

        # **使用 mapFromScene() 取得 scene 在 GraphicsView 中的真實寬度**
        scene_rect = scene.sceneRect()
        top_left = GraphicsView.mapFromScene(scene_rect.topLeft())
        top_right = GraphicsView.mapFromScene(scene_rect.topRight())
        real_scene_width = abs(top_right.x() - top_left.x())

        # print(f"Mapped scene width in GraphicsView: {real_scene_width}")

        # **更新 canvas_3d_view 的寬度**
        self.ui.canvas_3d_view.setMinimumWidth(int(real_scene_width))
        self.ui.canvas_3d_view.setMaximumWidth(int(real_scene_width))
        self.ui.canvas_3d_view.resize(int(real_scene_width), self.ui.canvas_3d_view.height())

        # print(f"Canvas3DView new width: {self.ui.canvas_3d_view.width()}")



    def setup_canvas_view(self):
        old_widget = self.ui.canvas_3d_view  # 取得 UI 內原本的 QWidget
        self.ui.canvas_3d_view = Canvas3DView(parent=self)  # ✅ 用 Canvas3DView 替換

        # **用 layout 替換舊的 widget**
        layout = old_widget.parentWidget().layout()
        layout.replaceWidget(old_widget, self.ui.canvas_3d_view)
        old_widget.deleteLater()  # 刪除原本的 QWidget

        # **確保 Layout 內沒有額外邊距**
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def keyPressEvent(self, event):
        key = event.text().lower()  # 用 event.text() 抓字元
        if key == 'd':
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        elif key == 'a':
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        else:
            super().keyPressEvent(event)

