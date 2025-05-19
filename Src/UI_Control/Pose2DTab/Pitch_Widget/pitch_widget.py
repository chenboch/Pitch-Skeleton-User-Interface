from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import sys
import os
from Pose2DTab.Pitch_Widget.pitch_ui import Ui_Pitch_UI
from datetime import datetime
from utils.timer import Timer
from cv_utils.cv_control import Camera, VideoLoader
from utils.selector import PersonSelector, KptSelector
from utils.analyze import PoseAnalyzer, JointAreaChecker
from ui_utils.table_control import KeypointTable
from ui_utils.graphicview_control import frame_view
import cv2
from UI_Control.vis_utils.vis_graph import GraphPlotter
from UI_Control.vis_utils.vis_image import ImageDrawer
from skeleton.detect_skeleton import PoseEstimater
import pyqtgraph as pg
from utils.model import Model

class PosePitchTabControl(QWidget):
    def __init__(self, model:Model, parent=None):
        super().__init__(parent)
        self.ui = Ui_Pitch_UI()
        self.ui.setupUi(self)
        self.model = model
        self.setup_components()
        self.init_var()
        self.bind_ui()

    def init_var(self):
        """Initialize variables and timer."""
        self.is_video = True if self.camera is None else False
        self.view_scene = QGraphicsScene()
        self.curve_scene = QGraphicsScene()
        self.view_scene.clear()
        self.curve_scene.clear()
        self.is_pitching = False
        self.initVideoVar()

    def initVideoVar(self):
        self.is_play = False
        self.correct_kpt_idx = 0
        self.is_processed = False
        self.play_times = 2
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

    def setup_components(self):
        self.camera = Camera()
        self.timer = QTimer()
        self.timer.timeout.connect(self.analyze_frame)

        self.countdown_timer = None
        self.record_checker = None
        self.record_timer = None
        self.pose_estimater = PoseEstimater(self.model)
        self.pose_analyzer = PoseAnalyzer(self.pose_estimater)
        self.graph_plotter = GraphPlotter(self.pose_analyzer)
        self.image_drawer = ImageDrawer(self.pose_estimater, self.pose_analyzer)
        self.video_loader = VideoLoader(self.image_drawer)

    def resize_event(self, event):
        new_size = event.size()
        print(f"PoseCameraTabControl resized to: {new_size.width()}x{new_size.height()}")
        # 在此執行你想要的操作
        if self.video_loader.video_name is not None:
            self.update_frame(frame_num=self.ui.frame_slider.value())
        super().resize_event(event)

    def bind_ui(self):
        """Bind UI element to their corresponding functions."""
        self.ui.cameraIdInput.valueChanged.connect(self.change_camera)
        self.ui.pitchInput.currentIndexChanged.connect(self.changePitcher)
        self.bindVideoUI()
        self.bindCheckBox()

    def bindVideoUI(self):
        self.kpt_table = KeypointTable(self.ui.kpt_table, self.pose_estimater)
        self.ui.kpt_table.cellActivated.connect(self.kpt_table.onCellClicked)
        self.frame_view = frame_view(self.ui.frame_view, self.view_scene, self.video_loader)
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame)

        self.ui.frame_view.mouse_press_event = self.mouse_press_event

    def bindCheckBox(self):
        """Bind UI CheckBox to their corresponding functions."""
        self.ui.cameraCheckBox.stateChanged.connect(self.toggle_camera)
        self.ui.recordCheckBox.stateChanged.connect(self.toggle_record)
        self.ui.select_checkbox.stateChanged.connect(self.toggle_select)
        self.ui.show_skeleton_checkbox.stateChanged.connect(self.toggle_show_skeleton)
        self.ui.select_kpt_checkbox.stateChanged.connect(self.toggle_kpt_select)
        self.ui.show_angle_checkbox.stateChanged.connect(self.toggle_show_angle_info)
        self.ui.show_bbox_checkbox.stateChanged.connect(self.toggle_show_bbox)
        self.ui.showLineCheckBox.stateChanged.connect(self.toggle_showgrid)
        self.ui.startPitchCheckBox.stateChanged.connect(self.togglePitching)

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

    def play_frame(self, start_num:int=0):
        for i in range(start_num, self.video_loader.total_frames):
            if not self.is_play:
                break
            self.ui.frame_slider.setValue(i)
            if i == self.video_loader.total_frames - 1 and self.is_play:
                self.play_btn_clicked()
            cv2.waitKey(15)

    def videoSilder(self, visible:bool):
        elements = [
            self.ui.back_key_btn,
            self.ui.play_btn,
            self.ui.forward_key_btn,
            self.ui.frame_slider,
            self.ui.frame_num_label,
            self.ui.curve_view
        ]

        for element in elements:
            element.setVisible(visible)

    def changePitcher(self):
        """Change the pitcher based on input value. 9: "左腕", 10: "右腕","""
        kpt_id = 10 if self.ui.pitchInput.currentIndex() == 0 else 9
        self.pose_estimater.setKptId(kpt_id)
        self.pose_estimater.setPitchHandId(kpt_id)

    def toggle_camera(self, state:int):
        """Toggle the camera on/off based on checkbox state."""
        if state == 2:
            if self.is_play:
                self.ui.cameraCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法開啟相機", "請先暫停播放影片")
                return
            self.reset()
            self.ui.show_angle_checkbox.setCheckState(0)
            self.ui.select_checkbox.setCheckState(0)
            self.ui.select_kpt_checkbox.setCheckState(0)
            self.is_video = False
            frame_width, frame_height, fps = self.camera.toggle_camera(True)
            self.model.setImageSize((frame_width, frame_height))
            self.ui.resolution_label.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
            self.timer.start(1)
            self.videoSilder(visible=False)
        else:
            if self.camera is not None:
                self.camera.toggle_camera(False)
            if self.timer is not None:
                self.timer.stop()

            self.view_scene.clear()
            self.curve_scene.clear()
            self.is_video = True
            self.videoSilder(visible=True)

    def togglePitching(self, state):
        if state == 2:
            if not self.ui.cameraCheckBox.isChecked():
                self.ui.startPitchCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法開始投球模式", "請先開啟相機")
                return
            if not self.ui.show_skeleton_checkbox.isChecked():
                self.ui.show_skeleton_checkbox.setCheckState(2)

            self.is_pitching = True
            self.record_checker = JointAreaChecker(self.camera.frame_size)
        else:
            self.is_pitching = False

    def toggle_record(self, state:int):
        """Start or stop video recording."""
        if state == 2:
            self.start_recording()
            print("record!!")
        else:
            if self.camera is None:
                return
            print("stop record!!")
            self.camera.stop_recording()

    def start_recording(self):
        """Start recording the video."""
        if self.camera is None:
            self.ui.recordCheckBox.setCheckState(0)
            return
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f'../../Db/Record/C{self.ui.cameraIdInput.value()}_Fps120_{current_time}'
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.join(output_dir, f'C{self.ui.cameraIdInput.value()}_Fps120_{current_time}.mp4')
        self.camera.start_recording(video_filename)

    def toggle_select(self, state:int):
        """Select a person based on checkbox state."""
        if state == 2:
            if not self.ui.show_skeleton_checkbox.isChecked():
                self.ui.select_checkbox.setCheckState(0)
                QMessageBox.warning(self, "無法選擇人", "請選擇顯示人體骨架!")
                return
            self.person_selector = PersonSelector()
            frame_num = self.ui.frame_slider.value() if self.is_video else None
            search_person_df = self.pose_estimater.get_person_df(frame_num=frame_num) if frame_num is not None else self.pose_estimater.pre_person_df
            self.person_selector.select(search_person_df = search_person_df)
            self.pose_estimater.setPersonId(self.person_selector.selected_id)
        else:
            self.pose_estimater.setPersonId(None)
            self.person_selector = None

    def toggle_kpt_select(self, state:int):
        """Toggle keypoint selection and trajectory visualization."""
        if state ==2:
            if not self.ui.select_checkbox.isChecked():
                self.ui.select_kpt_checkbox.setCheckState(0)
                QMessageBox.warning(self, "無法選擇關節點", "請選擇人!")
                return
            self.kpt_selector = KptSelector()
            self.pose_estimater.setKptId(10)
            self.image_drawer.set_show_traj(True)
        else:
            self.kpt_selector = None
            self.pose_estimater.setKptId(None)
            self.pose_estimater.clear_keypoint_buffer()
            self.image_drawer.set_show_traj(False)

    def toggle_show_skeleton(self, state:int):
        """Toggle skeleton detection and FPS control."""
        print("skeleton: "+ str(state))
        if state == 2 and self.ui.recordCheckBox.isChecked():
            self.ui.show_skeleton_checkbox.setCheckState(0)
            QMessageBox.warning(self, "無法選擇人", "請選擇顯示人體骨架!")
            return
        is_checked = state == 2
        if not is_checked:
            self.ui.show_angle_checkbox.setCheckState(0)
            self.ui.select_kpt_checkbox.setCheckState(0)
            self.ui.select_checkbox.setCheckState(0)

        self.pose_estimater.setDetect(is_checked)
        self.image_drawer.set_show_skeleton(is_checked)
        if self.camera is not None and not self.is_video:
            self.camera.set_fps_control(15 if is_checked else 1)

    def toggle_show_bbox(self, state:int):
        """Toggle bounding box visibility."""
        self.image_drawer.setShowBbox(state == 2)

    def toggle_show_angle_info(self, state:int):
        if not self.ui.select_checkbox.isChecked():
            self.ui.show_angle_checkbox.setCheckState(0)
            QMessageBox.warning(self, "無法顯示關節點角度資訊", "請選擇人!")
            return
        if state == 2:
            self.image_drawer.set_show_angle_info(True)
        else:
            self.image_drawer.set_show_angle_info(False)

    def toggle_showgrid(self, state:int):
        """Toggle gridline visibility."""
        self.image_drawer.set_show_grid(state == 2)

    def change_camera(self):
        """Change the camera based on input value."""
        if self.camera is not None:
            self.camera.set_camera_id(self.ui.cameraIdInput.value())

    def analyze_frame(self):
        """Analyze and process each frame from the camera or video"""
        fps = 0
        if self.is_video:
            frame_num = self.ui.frame_slider.value()
            self.ui.frame_num_label.setText(f'{frame_num}/{len(self.video_loader.video_frames) - 1}')
            frame = self.video_loader.get_video_image(frame_num)
            fps= self.pose_estimater.detect_keypoints(frame, frame_num, is_video=True)
            if self.pose_estimater.track_id is not None:
                self.pose_analyzer.addAnalyzeInfo(frame_num)
                self.graph_plotter.updateGraph(frame_num)
                self.kpt_table.importDataToTable(frame_num)

            self.update_frame(frame_num=frame_num)

            if frame_num == self.video_loader.total_frames - 1:
                self.handleVideoEnd()

        else:
            if not self.camera.frame_buffer.empty():
                frame = self.camera.frame_buffer.get().copy()
                fps = self.pose_estimater.detect_keypoints(frame, is_video=False)
                if self.is_pitching:
                    if self.ui.startPitchCheckBox.isChecked() and not self.ui.recordCheckBox.isChecked() and self.pose_estimater.track_id is None:
                        self.ui.select_checkbox.setCheckState(0)
                        self.ui.select_checkbox.setCheckState(2)
                    self.pitherAnaylze()
                self.update_frame(frame=frame)

        self.ui.fps_info_label.setText(f"{fps:02d}")

    def handleVideoEnd(self):
        """Handle the logic when video reaches its end."""
        self.play_times -= 1
        self.video_loader.save_video()

        if self.play_times > 0:
            # Replay the video
            self.play_btn_clicked()
            self.ui.frame_slider.setValue(0)
            self.play_btn_clicked()
        else:
            # Stop playback and reset
            self.play_btn_clicked()
            self.ui.cameraCheckBox.setCheckState(2)
            self.ui.show_skeleton_checkbox.setCheckState(0)
            self.ui.show_skeleton_checkbox.setCheckState(2)
            self.ui.select_checkbox.setCheckState(2)
            self.ui.startPitchCheckBox.setCheckState(2)
            self.play_times = 2

    def pitherAnaylze(self):
        if self.record_checker is not None:
            pos = self.pose_estimater.getPrePersonDf()
            if self.pose_estimater.setPersonId is not None:
                if self.record_checker.is_joint_in_area(pos):
                    self.initCountdownTimer(2)
        else:
            self.initRecorderTimer(2)

    def initCountdownTimer(self, duration:int):
        if self.countdown_timer is None:
            self.countdown_timer = Timer(duration)
            self.countdown_timer.start()

    def initRecorderTimer(self, duration:int):
        if self.record_timer is None:
            self.ui.show_bbox_checkbox.setCheckState(0)
            self.ui.select_kpt_checkbox.setCheckState(0)
            self.ui.select_checkbox.setCheckState(0)
            self.ui.show_skeleton_checkbox.setCheckState(0)
            self.ui.recordCheckBox.setCheckState(2)
            self.record_timer = Timer(duration)
            self.record_timer.start()

    def update_frame(self, frame: np.ndarray = None, frame_num:int = None):
        """Update the displayed frame with additional analysis."""
        # 更新當前的frame和frame_num
        if self.is_video and frame_num is not None:
            frame = self.video_loader.get_video_image(frame_num)
        countdown_time = self.updateTimers()
        drawed_img = self.image_drawer.drawInfo(frame, frame_num, self.pose_estimater.kpt_buffer, countdown_time)
        self.show_image(drawed_img, self.view_scene, self.ui.frame_view)
        self.graph_plotter.resize_graph(self.ui.curve_view.width(),self.ui.curve_view.height())

    def updateTimers(self):
        countdown_time = None
        if self.countdown_timer is not None:
            countdown_time = self.countdown_timer.get_remaining_time()
            if countdown_time == 0:
                self.resetCountdownTimer()
        if self.record_timer is not None:
            countdown_time = self.record_timer.get_remaining_time()
            if countdown_time == 0:
                self.resetRecordTimer()
        return countdown_time

    def resetCountdownTimer(self):
        self.countdown_timer = None
        self.record_checker = None

    def resetRecordTimer(self):
        self.record_timer = None
        video_path = self.camera.video_path
        self.video_loader.reset()
        self.ui.recordCheckBox.setCheckState(0)
        self.ui.startPitchCheckBox.setCheckState(0)
        self.ui.cameraCheckBox.setCheckState(0)
        self.video_loader.load_video(video_path)
        self.check_video_load()

    def check_video_load(self):
        """檢查影片是否讀取完成，並更新 UI 元素。"""
        # 檢查是否有影片名稱，若無則不執行後續操作
        if not self.video_loader.video_name:
            return
        # 若影片正在讀取中，定時檢查讀取狀況
        if self.video_loader.is_loading:
            # self.ui.video_name_label.setText("讀取影片中")
            QTimer.singleShot(100, self.check_video_load)  # 每100ms 檢查一次
            return
        # 影片讀取完成後更新 UI 元素
        self.update_video_info()
        self.initanalyze_frame()

    def update_video_info(self):
        """更新與影片相關的資訊顯示在 UI 上。"""
        self.reset()
        self.init_frame_slider()
        self.init_graph()
        self.update_frame(frame_num=0)
        self.model.setImageSize(self.video_loader.video_size)
        video_size = self.video_loader.video_size
        self.ui.resolution_label.setText(f"(0,0) - {video_size[0]} x {video_size[1]}")

    def initanalyze_frame(self):
        self.ui.show_skeleton_checkbox.setCheckState(2)
        frame = self.video_loader.get_video_image(0)
        fps = self.pose_estimater.detect_keypoints(frame, 0, is_video=True)
        self.ui.select_checkbox.setCheckState(2)
        self.ui.select_kpt_checkbox.setCheckState(2)
        self.ui.show_angle_checkbox.setCheckState(2)
        self.ui.play_btn.click()

    def init_frame_slider(self):
        """初始化影片滑桿和相關的標籤。"""
        total_frames = self.video_loader.total_frames
        self.ui.frame_slider.setMinimum(0)
        self.ui.frame_slider.setMaximum(total_frames - 1)
        self.ui.frame_slider.setValue(0)
        self.ui.frame_num_label.setText(f'0/{total_frames - 1}')

    def resetframe_slider(self):
        self.ui.frame_slider.setValue(0)
        self.ui.frame_slider.setRange(0, 0)

    def init_graph(self):
        """初始化圖表和模型設定。"""
        total_frames = self.video_loader.total_frames
        print("video frames: "+ str(total_frames))
        self.graph_plotter._init_graph(total_frames)
        self.show_graph(self.curve_scene, self.ui.curve_view)

    def show_graph(self, scene:QGraphicsScene, graphicview:QGraphicsView):
        scene.clear()
        graph = self.graph_plotter.graph
        graph.resize(graphicview.width(),graphicview.height())
        scene.addWidget(graph)
        graphicview.setScene(scene)
        graphicview.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def reset(self):
        self.pose_estimater.reset()
        self.pose_analyzer.reset()
        self.graph_plotter.reset()
        self.image_drawer.reset()
        self.resetframe_slider()
        self.view_scene.clear()
        self.curve_scene.clear()

    def show_image(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        """Display an image in the QGraphicsView."""
        scene.clear()
        if image is None:
            print(self.video_loader.total_frames)
            # print(self.video_loader.video_frames)
        h, w = image.shape[:2]
        qImg = QImage(image, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def mouse_press_event(self, event):
        """Handle mouse events for person and keypoint selection."""
        if not self.ui.frame_view.rect().contains(event.pos()):
            return

        scene_pos = self.ui.frame_view.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        search_person_df = self.pose_estimater.get_person_df(frame_num = self.ui.frame_slider.value()) if self.is_video else self.pose_estimater.pre_person_df

        if self.ui.select_checkbox.isChecked() and not self.kpt_table.label_kpt :
            if event.button() == Qt.LeftButton:
                self.person_selector.select(x, y, search_person_df)
                self.pose_estimater.setPersonId(self.person_selector.selected_id)

        if self.ui.select_kpt_checkbox.isChecked() and not self.kpt_table.label_kpt :
            if event.button() == Qt.LeftButton:
                self.kpt_selector.select(x, y, search_person_df)
                self.pose_estimater.setKptId(self.kpt_selector.selected_id)

        if self.kpt_table.label_kpt:
            if event.button() == Qt.LeftButton:
                self.kpt_table.sendToTable(x, y, 1, self.ui.frame_slider.value())
            elif event.button() == Qt.RightButton:
                self.kpt_table.sendToTable(0, 0, 0, self.ui.frame_slider.value())

        if self.is_video:
            self.update_frame(frame_num=self.ui.frame_slider.value())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PosePitchTabControl()
    window.show()
    sys.exit(app.exec_())
