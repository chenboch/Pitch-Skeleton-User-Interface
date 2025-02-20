import sys
import PyQt5.QtOpenGL  # 這一行最關鍵
from vispy import scene
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
# from UI_Control.Pose2DTab.Camera_Widget.camera_widget import PoseCameraTabControl
from UI_Control.Pose2DTab.Video_Widget.video_widget import PoseVideoTabControl as Pose2DVideoTabControl
# # from UI_Control.Pose2DTab.Pitch_Widget.pitch_widget import PosePitchTabControl
from UI_Control.Pose3DTab.Video_Widget.video_widget import PoseVideoTabControl as Pose3DVideoTabControl
from UI_Control.Main_UI.main_window import Ui_MainWindow
from skeleton.model.wrapper import Wrapper
import logging
import sys
import os


class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # setting camera
        self.wrapper = Wrapper(track_model="ByteTracker", pose_model="vit-pose")
        self.init_tabs()

    def init_tabs(self):
        # self.camera_tab = PoseCameraTabControl(self.wrapper)
        # self.ui.Two_d_Tab.addTab(self.camera_tab, "2D 相機")
        self.video_tab = Pose2DVideoTabControl(self.wrapper)
        self.ui.Two_d_Tab.addTab(self.video_tab, "2D 影片")
        # self.pitch_tab = PosePitchTabControl(self.wrapper)
        # self.ui.Two_d_Tab.addTab(self.pitch_tab, "2D 投手")
        self.video3d_tab = Pose3DVideoTabControl(self.wrapper)
        self.ui.Two_d_Tab.addTab(self.video3d_tab, "3D 影片")




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    app = QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())

