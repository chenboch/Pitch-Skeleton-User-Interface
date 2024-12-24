from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt5.QtCore import QTimer
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import random

class Skeleton3DViewer(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.joint_positions = self.generate_random_skeleton()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_skeleton)
        self.timer.start(100)  # 每100毫秒刷新一次

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2, 0.3, 0.3, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5)  # 往後移動攝像機

        # 繪製骨架
        glColor3f(1.0, 0.0, 0.0)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for (start, end) in self.get_skeleton_connections():
            glVertex3fv(self.joint_positions[start])
            glVertex3fv(self.joint_positions[end])
        glEnd()

    def generate_random_skeleton(self):
        # 生成隨機骨架數據 (10個關節點)
        return {i: (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(10)}

    def get_skeleton_connections(self):
        # 定義骨架連接 (假設一些連接關係)
        return [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]

    def update_skeleton(self):
        # 模擬每幀更新數據 (你可以替換為實際的骨架數據)
        self.joint_positions = self.generate_random_skeleton()
        self.update()  # 觸發 repaint

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Skeleton Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.viewer = Skeleton3DViewer()
        self.setCentralWidget(self.viewer)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
