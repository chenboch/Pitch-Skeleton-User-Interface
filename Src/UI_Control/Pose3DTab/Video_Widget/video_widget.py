from UI_Control.BaseTab import BasePoseVideoTab
from UI_Control.ui_utils import KeypointTable
from skeleton import VidePose3DEstimater
from .video_ui import Ui_video_widget
from .canvas3d_widget import Canvas3DView

class PoseVideoTabControl(BasePoseVideoTab):
    def __init__(self, wrapper, parent = None):
        super().__init__(wrapper, parent)
        self.ui = Ui_video_widget()
        self.ui.setupUi(self)
        self.bind_ui()


    def bind_ui(self):
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


    def toggle_detect(self):
        self.ui.show_skeleton_checkbox.setChecked(True)
        frame = self.video_loader.get_video_image(0)
        fps = self.pose_estimater.detect_keypoints(frame, 0)
        self.ui.play_btn.click()

    # def setup_canvas_view(self):
    #     old_widget = self.ui.canvas_3d_view  # 取得 UI 內原本的 QWidget
    #     self.ui.canvas_3d_view = Canvas3DView()  # 替換為 Canvas3DView
    #     layout = self.ui.verticalLayout_3
    #     layout.replaceWidget(old_widget, self.ui.canvas_3d_view)
    #     old_widget.deleteLater()


    def setup_pose_estimater(self):
        """Setup 2D pose estimator."""
        self.pose_estimater = VidePose3DEstimater(self.wrapper)


    def keyPressEvent(self, event):
        key = event.text().lower()  # 用 event.text() 抓字元
        if key == 'd':
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        elif key == 'a':
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        else:
            super().keyPressEvent(event)

