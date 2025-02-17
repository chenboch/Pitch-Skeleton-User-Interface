from UI_Control.BaseTab import BasePoseVideoTab
from UI_Control.ui_utils import KeypointTable
from skeleton import VidePose2DEstimater
from .video_ui import Ui_video_widget



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

    def setup_pose_estimater(self):
        """Setup 2D pose estimator."""
        self.pose_estimater = VidePose2DEstimater(self.wrapper)
        self.model_name = self.wrapper.pose2d_estimator.model_name

    def key_press_event(self, event):
        key = event.key()
        slider = self.ui.frame_slider

        if chr(key).lower() == 'd':  # 按下 'D' 或 'd'
            new_value = slider.value() + 1
            slider.setValue(min(new_value, slider.maximum()))  # 防止超过最大值
        elif chr(key).lower() == 'a':  # 按下 'A' 或 'a'
            new_value = slider.value() - 1
            slider.setValue(max(new_value, slider.minimum()))  # 防止低于最小值
        else:
            super().keyPressEvent(event)  # 调用父类默认行为

