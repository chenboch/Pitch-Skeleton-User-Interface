from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

from skeleton.datasets import coco_keypoint_info
from vispy.app import use_app
import numpy as np
import sys
import os
from vispy import app as vis_app, visuals, scene, geometry, color
from vispy.scene.cameras import ArcballCamera, MagnifyCamera, perspective, turntable
from vispy.visuals import transforms
from vispy.color import Color
import polars as pl
from vispy.scene.visuals import XYZAxis

# CANVAS_SIZE = (1920, 300)  # (width, height)

class Canvas3DView(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        # self.ui = Ui_canvas_3d_view()
        # self.ui.setupUi(self)
        self.lines_plot = []
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=True, bgcolor=Color("#F2F2F2", alpha=0.1))
        self.view = self.canvas.central_widget.add_view()  # 添加视图
        # setting camera
         # **關鍵！將 VisPy 畫布加到 QWidget 的 Layout**
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas.native)  # 讓 VisPy 繪製區域顯示在 PyQt Widget 內
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.view.camera = scene.TurntableCamera(
            elevation=0, azimuth=0, roll=0, up="+z", distance=10,translate_speed=100,
        )
        self.axis = XYZAxis(parent=self.view.scene)
        self.plane = self.set_floor_plane()
        self.set_lines_plot()
        self.cx = 0
        self.cy = 0
        self.cz = 0

    def set_floor_plane(
        self,
        XL=6,
        YL=6,
        WS=6,
        HS=6,
        D="+z",
        translate=(0, -1, -1),
        # scale=(1.0, 1.0, 1.0),
    ):
        vertices, faces, outline = geometry.create_plane(
            width=XL, height=YL, width_segments=WS, height_segments=HS, direction=D
        )
        colors = []
        for _ in range(faces.shape[0]):
            colors.append(np.array([0, 0, 0, 0.1]))
        plane = scene.visuals.Plane(
            width=XL,
            height=YL,
            width_segments=WS,
            height_segments=HS,
            direction=D,
            face_colors=np.array(colors),
            edge_color=color.color_array.Color(
                color="white", alpha=None, clip=False),
            parent=self.view.scene,
        )
        plane.transform = transforms.STTransform(
            translate= translate
        )
            # translate=translate, scale=scale)

        return plane

    def set_datas(self, data, image_size=(1920, 1080), rfoot=3, lfoot=6, D="+z"):
        data = np.array(data)
        w, h = image_size
        # 更新中心點座標
        data[...,0] = - data[...,0]
        data[...,0] = data[...,0]
        data[...,1] = data[...,1]
        data[...,2] = data[...,2]
        w = np.max(data[..., 0]) - np.min(data[..., 0])
        h = np.max(data[..., 1]) - np.min(data[..., 1])
        d = np.max(data[..., 2]) - np.min(data[..., 2])
        cx = np.mean(data[..., 0])  # X 軸中心
        cy = np.mean(data[..., 1])  # Y 軸中心
        cz = np.mean(data[..., 2])  # Z 軸中心
        print(f"{cx}, {cy}, {cz}")

        # 計算地板高度
        rmax = np.sort(data[:, rfoot, 2])[:]
        lmax = np.sort(data[:, lfoot, 2])[:]
        floor = np.average(np.squeeze(
            np.concatenate(([rmax], [lmax]), axis=0)))
        floor = np.round(floor, 2)
        print("Floor:", floor)
        # 攝影機向左平移
        self.view.camera.center = (cx, cy, cz)

        self.view.camera.translate_speed = max(w // 50, 100)
        # 移除舊的平面，創建新平面並平移
        self.view.scene._remove_child(self.plane)
        vertices, faces, outline = geometry.create_plane(
            width=w, height=h, width_segments=4, height_segments=4, direction=D
        )
        colors = []
        for _ in range(faces.shape[0]):
            colors.append(np.array([0, 0, 0, 0.1]))
        plane = scene.visuals.Plane(
            width=w,
            height=h,
            width_segments=4,
            height_segments=4,
            direction=D,
            face_colors=np.array(colors),
            edge_color=color.color_array.Color(
                color="white", alpha=None, clip=False),
            parent=self.view.scene,
        )

        # 更新物件的中心點
        self.cx = cx
        self.cy = cy
        self.cz = cz

    def set_lines_plot(self):
        for _ in range(len(coco_keypoint_info["keypoints"])):
            plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
            self.lines_plot.append(plot3D(parent=self.view.scene))
        plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)

    def update_points(self, pos, rfoot=3, lfoot=6):
        pos = np.array(pos)
        pos[:, 0] = -pos[:, 0]
        # pos[:, 1] -= 1  # 向下移動 y
        # pos[:, 2] -= 1  # 向下移動 z
        pos[:, 0] *= 5
        pos[:, 1] *= 5 # 向下移動 y
        pos[:, 2] *= 5  # 向下移動 z
        # pos[:, 1] -= 1  # 向下移動 y
        pos[:, 2] -= 1 # 向下移動 z

        # 計算新的中心點
        cx = np.mean(pos[:, 0])  # X 軸中心
        cy = np.mean(pos[:, 1])  # Y 軸中心
        cz = np.min(pos[:, 2])  # Z 軸中心

        # **正確的刪除舊的 XYZ 軸**
        if hasattr(self, "axis") and self.axis is not None:
            self.axis.parent = None  # 這樣才是真正從場景中移除

        # **創建新的 XYZ 軸，並移動到新位置**
        self.axis = XYZAxis(parent=self.view.scene)
        self.axis.transform = transforms.STTransform(translate=(cx, cy, cz))
        self.plane.transform = transforms.STTransform(
            translate=(cx, cy, cz)
        )

        # print(f"Moving XYZ axis to: ({cx}, {cy}, {cz})")

        # **畫骨架線**
        for idx, line in enumerate(coco_keypoint_info["skeleton_links"]):
            limb_pos = []
            color = 'red'
            if line in coco_keypoint_info["right_limb"]:
                color = 'blue'
            elif line in coco_keypoint_info["left_limb"]:
                color = 'red'
            else:
                color = 'orange'

            for point in line:
                limb_pos.append(pos[point])

            self.lines_plot[idx].set_data(
                limb_pos,
                marker_size=7,
                width=5.0,
                color=color,
                edge_color="y",
                edge_width=2,
                symbol="x",
                face_color=(0.2, 0.2, 1, 0.8),
            )



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Canvas3DView()
    window.show()
    sys.exit(app.exec_())