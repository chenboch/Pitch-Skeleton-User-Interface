from UI_Control.BaseTab import BasePoseCameraTab
from UI_Control.ui_utils import KeypointTable
from skeleton import CameraPose2DEstimater
from .camera_ui import Ui_camera_widget



class PoseCameraTabControl(BasePoseCameraTab):
    def __init__(self, wrapper, parent = None):
        super().__init__(wrapper, parent)
        self.ui = Ui_camera_widget()
        self.ui.setupUi(self)
        self.bind_ui()

    def bind_ui(self):
        """Bind UI elements to their corresponding functions."""
        self.ui.camera_checkbox.stateChanged.connect(self.toggle_camera)
        self.ui.record_checkbox.stateChanged.connect(self.toggle_record)
        self.ui.select_checkbox.stateChanged.connect(self.toggle_select)
        self.ui.show_skeleton_checkbox.stateChanged.connect(self.toggle_show_skeleton)
        self.ui.select_kpt_checkbox.stateChanged.connect(self.toggle_kpt_select)
        self.ui.show_bbox_checkbox.stateChanged.connect(self.toggle_show_bbox)
        self.ui.show_line_checkbox.stateChanged.connect(self.toggle_showgrid)
        self.ui.camera_id.valueChanged.connect(self.change_camera)
        self.ui.frame_view.mousePressEvent = self.mouse_press_event

    def setup_pose_estimater(self):
        """Setup 2D pose estimator."""
        self.pose_estimater = CameraPose2DEstimater(self.wrapper)
        self.model_name = self.wrapper.pose2d_estimator.model_name

