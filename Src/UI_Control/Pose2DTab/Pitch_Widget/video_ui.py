# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\user\Desktop\Pitch-Skeleton-User-Interface\Src\UI_Control\Pose2DTab\Pitch_Widget\video.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_video_widget(object):
    def setupUi(self, video_widget):
        video_widget.setObjectName("video_widget")
        video_widget.resize(1200, 800)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        video_widget.setFont(font)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(video_widget)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.resolution_label = QtWidgets.QLabel(video_widget)
        self.resolution_label.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.resolution_label.setFont(font)
        self.resolution_label.setObjectName("resolution_label")
        self.verticalLayout_3.addWidget(self.resolution_label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.left_frame_view = QtWidgets.QGraphicsView(video_widget)
        self.left_frame_view.setMinimumSize(QtCore.QSize(480, 360))
        self.left_frame_view.setObjectName("left_frame_view")
        self.horizontalLayout.addWidget(self.left_frame_view)
        self.right_frame_view = QtWidgets.QGraphicsView(video_widget)
        self.right_frame_view.setMinimumSize(QtCore.QSize(480, 360))
        self.right_frame_view.setObjectName("right_frame_view")
        self.horizontalLayout.addWidget(self.right_frame_view)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.back_key_btn = QtWidgets.QPushButton(video_widget)
        self.back_key_btn.setMinimumSize(QtCore.QSize(50, 30))
        self.back_key_btn.setMaximumSize(QtCore.QSize(50, 30))
        self.back_key_btn.setObjectName("back_key_btn")
        self.horizontalLayout_3.addWidget(self.back_key_btn)
        self.play_btn = QtWidgets.QPushButton(video_widget)
        self.play_btn.setMinimumSize(QtCore.QSize(50, 30))
        self.play_btn.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.play_btn.setFont(font)
        self.play_btn.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.play_btn.setIconSize(QtCore.QSize(20, 20))
        self.play_btn.setObjectName("play_btn")
        self.horizontalLayout_3.addWidget(self.play_btn)
        self.forward_key_btn = QtWidgets.QPushButton(video_widget)
        self.forward_key_btn.setMinimumSize(QtCore.QSize(50, 30))
        self.forward_key_btn.setMaximumSize(QtCore.QSize(50, 30))
        self.forward_key_btn.setObjectName("forward_key_btn")
        self.horizontalLayout_3.addWidget(self.forward_key_btn)
        self.frame_slider = QtWidgets.QSlider(video_widget)
        self.frame_slider.setMinimumSize(QtCore.QSize(300, 30))
        self.frame_slider.setOrientation(QtCore.Qt.Horizontal)
        self.frame_slider.setObjectName("frame_slider")
        self.horizontalLayout_3.addWidget(self.frame_slider)
        self.frame_num_label = QtWidgets.QLabel(video_widget)
        self.frame_num_label.setMinimumSize(QtCore.QSize(20, 20))
        self.frame_num_label.setObjectName("frame_num_label")
        self.horizontalLayout_3.addWidget(self.frame_num_label)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3.setStretch(1, 2)
        self.horizontalLayout_5.addLayout(self.verticalLayout_3)
        self.tab_widget = QtWidgets.QTabWidget(video_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab_widget.sizePolicy().hasHeightForWidth())
        self.tab_widget.setSizePolicy(sizePolicy)
        self.tab_widget.setMinimumSize(QtCore.QSize(400, 0))
        self.tab_widget.setMaximumSize(QtCore.QSize(450, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tab_widget.setFont(font)
        self.tab_widget.setObjectName("tab_widget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.setting_groupbox = QtWidgets.QGroupBox(self.tab)
        self.setting_groupbox.setMinimumSize(QtCore.QSize(450, 625))
        self.setting_groupbox.setMaximumSize(QtCore.QSize(450, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.setting_groupbox.setFont(font)
        self.setting_groupbox.setObjectName("setting_groupbox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.setting_groupbox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.file_groupbox = QtWidgets.QGroupBox(self.setting_groupbox)
        self.file_groupbox.setObjectName("file_groupbox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.file_groupbox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.name_label = QtWidgets.QLabel(self.file_groupbox)
        self.name_label.setObjectName("name_label")
        self.horizontalLayout_6.addWidget(self.name_label)
        self.video_name_label = QtWidgets.QLabel(self.file_groupbox)
        self.video_name_label.setMinimumSize(QtCore.QSize(130, 0))
        self.video_name_label.setText("")
        self.video_name_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.video_name_label.setObjectName("video_name_label")
        self.horizontalLayout_6.addWidget(self.video_name_label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.fps_label = QtWidgets.QLabel(self.file_groupbox)
        self.fps_label.setObjectName("fps_label")
        self.horizontalLayout_2.addWidget(self.fps_label)
        self.fps_info_label = QtWidgets.QLabel(self.file_groupbox)
        self.fps_info_label.setMinimumSize(QtCore.QSize(130, 0))
        self.fps_info_label.setText("")
        self.fps_info_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.fps_info_label.setObjectName("fps_info_label")
        self.horizontalLayout_2.addWidget(self.fps_info_label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.load_original_video_btn = QtWidgets.QPushButton(self.file_groupbox)
        self.load_original_video_btn.setMinimumSize(QtCore.QSize(150, 30))
        self.load_original_video_btn.setObjectName("load_original_video_btn")
        self.gridLayout.addWidget(self.load_original_video_btn, 1, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.verticalLayout.addWidget(self.file_groupbox)
        self.file_groupbox_2 = QtWidgets.QGroupBox(self.setting_groupbox)
        self.file_groupbox_2.setObjectName("file_groupbox_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.file_groupbox_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.name_label_2 = QtWidgets.QLabel(self.file_groupbox_2)
        self.name_label_2.setObjectName("name_label_2")
        self.horizontalLayout_7.addWidget(self.name_label_2)
        self.video_name_label_2 = QtWidgets.QLabel(self.file_groupbox_2)
        self.video_name_label_2.setMinimumSize(QtCore.QSize(130, 0))
        self.video_name_label_2.setText("")
        self.video_name_label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.video_name_label_2.setObjectName("video_name_label_2")
        self.horizontalLayout_7.addWidget(self.video_name_label_2)
        self.verticalLayout_6.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.fps_label_2 = QtWidgets.QLabel(self.file_groupbox_2)
        self.fps_label_2.setObjectName("fps_label_2")
        self.horizontalLayout_8.addWidget(self.fps_label_2)
        self.fps_info_label_2 = QtWidgets.QLabel(self.file_groupbox_2)
        self.fps_info_label_2.setMinimumSize(QtCore.QSize(130, 0))
        self.fps_info_label_2.setText("")
        self.fps_info_label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.fps_info_label_2.setObjectName("fps_info_label_2")
        self.horizontalLayout_8.addWidget(self.fps_info_label_2)
        self.verticalLayout_6.addLayout(self.horizontalLayout_8)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.load_original_video_btn_2 = QtWidgets.QPushButton(self.file_groupbox_2)
        self.load_original_video_btn_2.setMinimumSize(QtCore.QSize(150, 30))
        self.load_original_video_btn_2.setObjectName("load_original_video_btn_2")
        self.gridLayout_2.addWidget(self.load_original_video_btn_2, 1, 0, 1, 1)
        self.verticalLayout_6.addLayout(self.gridLayout_2)
        self.verticalLayout.addWidget(self.file_groupbox_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.verticalLayout_5.addWidget(self.setting_groupbox)
        self.tab_widget.addTab(self.tab, "")
        self.horizontalLayout_5.addWidget(self.tab_widget)

        self.retranslateUi(video_widget)
        self.tab_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(video_widget)

    def retranslateUi(self, video_widget):
        _translate = QtCore.QCoreApplication.translate
        video_widget.setWindowTitle(_translate("video_widget", "Form"))
        self.resolution_label.setText(_translate("video_widget", "(0, 0) - "))
        self.back_key_btn.setText(_translate("video_widget", "<<"))
        self.play_btn.setText(_translate("video_widget", "▶︎"))
        self.forward_key_btn.setText(_translate("video_widget", ">>"))
        self.frame_num_label.setText(_translate("video_widget", "0/0"))
        self.setting_groupbox.setTitle(_translate("video_widget", "2D 關節點"))
        self.file_groupbox.setTitle(_translate("video_widget", "左側檔案"))
        self.name_label.setText(_translate("video_widget", "檔名:"))
        self.fps_label.setText(_translate("video_widget", "FPS:"))
        self.load_original_video_btn.setText(_translate("video_widget", "載入影片"))
        self.file_groupbox_2.setTitle(_translate("video_widget", "右側檔案"))
        self.name_label_2.setText(_translate("video_widget", "檔名:"))
        self.fps_label_2.setText(_translate("video_widget", "FPS:"))
        self.load_original_video_btn_2.setText(_translate("video_widget", "載入影片"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab), _translate("video_widget", "一般設定"))
