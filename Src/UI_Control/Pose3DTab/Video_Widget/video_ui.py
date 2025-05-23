# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\user\Desktop\Pitch-Skeleton-User-Interface\Src\UI_Control\Pose3DTab\Video_Widget\video.ui'
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
        font.setPointSize(12)
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
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setSpacing(3)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.frame_view = QtWidgets.QGraphicsView(video_widget)
        self.frame_view.setMinimumSize(QtCore.QSize(480, 360))
        self.frame_view.setObjectName("frame_view")
        self.verticalLayout_7.addWidget(self.frame_view)
        self.canvas_3d_view = QtWidgets.QWidget(video_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.canvas_3d_view.sizePolicy().hasHeightForWidth())
        self.canvas_3d_view.setSizePolicy(sizePolicy)
        self.canvas_3d_view.setMinimumSize(QtCore.QSize(300, 300))
        self.canvas_3d_view.setObjectName("canvas_3d_view")
        self.verticalLayout_7.addWidget(self.canvas_3d_view)
        self.verticalLayout_7.setStretch(0, 1)
        self.verticalLayout_7.setStretch(1, 1)
        self.verticalLayout_3.addLayout(self.verticalLayout_7)
        self.curve_view = QtWidgets.QGraphicsView(video_widget)
        self.curve_view.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.curve_view.setObjectName("curve_view")
        self.verticalLayout_3.addWidget(self.curve_view)
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
        self.load_processed_video_btn = QtWidgets.QPushButton(self.file_groupbox)
        self.load_processed_video_btn.setMinimumSize(QtCore.QSize(150, 30))
        self.load_processed_video_btn.setObjectName("load_processed_video_btn")
        self.gridLayout.addWidget(self.load_processed_video_btn, 2, 0, 1, 1)
        self.load_original_video_btn = QtWidgets.QPushButton(self.file_groupbox)
        self.load_original_video_btn.setMinimumSize(QtCore.QSize(150, 30))
        self.load_original_video_btn.setObjectName("load_original_video_btn")
        self.gridLayout.addWidget(self.load_original_video_btn, 1, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.verticalLayout.addWidget(self.file_groupbox)
        self.groupBox = QtWidgets.QGroupBox(self.setting_groupbox)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.start_code_btn = QtWidgets.QPushButton(self.groupBox)
        self.start_code_btn.setMinimumSize(QtCore.QSize(150, 30))
        self.start_code_btn.setObjectName("start_code_btn")
        self.verticalLayout_6.addWidget(self.start_code_btn)
        self.verticalLayout.addWidget(self.groupBox)
        self.display_groupbox = QtWidgets.QGroupBox(self.setting_groupbox)
        self.display_groupbox.setObjectName("display_groupbox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.display_groupbox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.show_skeleton_checkbox = QtWidgets.QCheckBox(self.display_groupbox)
        self.show_skeleton_checkbox.setChecked(False)
        self.show_skeleton_checkbox.setObjectName("show_skeleton_checkbox")
        self.gridLayout_2.addWidget(self.show_skeleton_checkbox, 0, 1, 1, 1)
        self.show_bbox_checkbox = QtWidgets.QCheckBox(self.display_groupbox)
        self.show_bbox_checkbox.setCheckable(True)
        self.show_bbox_checkbox.setChecked(False)
        self.show_bbox_checkbox.setObjectName("show_bbox_checkbox")
        self.gridLayout_2.addWidget(self.show_bbox_checkbox, 0, 2, 1, 1)
        self.select_checkbox = QtWidgets.QCheckBox(self.display_groupbox)
        self.select_checkbox.setObjectName("select_checkbox")
        self.gridLayout_2.addWidget(self.select_checkbox, 1, 1, 1, 1)
        self.select_kpt_checkbox = QtWidgets.QCheckBox(self.display_groupbox)
        self.select_kpt_checkbox.setObjectName("select_kpt_checkbox")
        self.gridLayout_2.addWidget(self.select_kpt_checkbox, 1, 2, 1, 1)
        self.show_angle_checkbox = QtWidgets.QCheckBox(self.display_groupbox)
        self.show_angle_checkbox.setObjectName("show_angle_checkbox")
        self.gridLayout_2.addWidget(self.show_angle_checkbox, 2, 1, 1, 1)
        self.verticalLayout.addWidget(self.display_groupbox)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.verticalLayout_5.addWidget(self.setting_groupbox)
        self.tab_widget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.id_adjust_groupbox = QtWidgets.QGroupBox(self.tab_2)
        self.id_adjust_groupbox.setObjectName("id_adjust_groupbox")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.id_adjust_groupbox)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.id_label = QtWidgets.QLabel(self.id_adjust_groupbox)
        self.id_label.setMinimumSize(QtCore.QSize(120, 0))
        self.id_label.setMaximumSize(QtCore.QSize(120, 16777215))
        self.id_label.setObjectName("id_label")
        self.horizontalLayout_4.addWidget(self.id_label)
        self.before_correct_id = QtWidgets.QSpinBox(self.id_adjust_groupbox)
        self.before_correct_id.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.before_correct_id.setObjectName("before_correct_id")
        self.horizontalLayout_4.addWidget(self.before_correct_id)
        self.label_3 = QtWidgets.QLabel(self.id_adjust_groupbox)
        self.label_3.setMaximumSize(QtCore.QSize(15, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_4.addWidget(self.label_3)
        self.after_correct_id = QtWidgets.QSpinBox(self.id_adjust_groupbox)
        self.after_correct_id.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.after_correct_id.setObjectName("after_correct_id")
        self.horizontalLayout_4.addWidget(self.after_correct_id)
        self.id_correct_btn = QtWidgets.QPushButton(self.id_adjust_groupbox)
        self.id_correct_btn.setMinimumSize(QtCore.QSize(100, 0))
        self.id_correct_btn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.id_correct_btn.setObjectName("id_correct_btn")
        self.horizontalLayout_4.addWidget(self.id_correct_btn)
        self.verticalLayout_8.addLayout(self.horizontalLayout_4)
        self.verticalLayout_4.addWidget(self.id_adjust_groupbox)
        self.kpt_adjust_groupbox = QtWidgets.QGroupBox(self.tab_2)
        self.kpt_adjust_groupbox.setObjectName("kpt_adjust_groupbox")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.kpt_adjust_groupbox)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.kpt_table = QtWidgets.QTableWidget(self.kpt_adjust_groupbox)
        self.kpt_table.setMinimumSize(QtCore.QSize(400, 0))
        self.kpt_table.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.kpt_table.setObjectName("kpt_table")
        self.kpt_table.setColumnCount(5)
        self.kpt_table.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.kpt_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.kpt_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.kpt_table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.kpt_table.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.kpt_table.setHorizontalHeaderItem(4, item)
        self.verticalLayout_9.addWidget(self.kpt_table)
        self.verticalLayout_4.addWidget(self.kpt_adjust_groupbox)
        self.tab_widget.addTab(self.tab_2, "")
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
        self.file_groupbox.setTitle(_translate("video_widget", "檔案"))
        self.name_label.setText(_translate("video_widget", "檔名:"))
        self.fps_label.setText(_translate("video_widget", "FPS:"))
        self.load_processed_video_btn.setText(_translate("video_widget", "載入處理過的影片"))
        self.load_original_video_btn.setText(_translate("video_widget", "載入原始影片"))
        self.groupBox.setTitle(_translate("video_widget", "2D3D Process"))
        self.start_code_btn.setText(_translate("video_widget", "處理和儲存"))
        self.display_groupbox.setTitle(_translate("video_widget", "顯示資訊"))
        self.show_skeleton_checkbox.setText(_translate("video_widget", "人體骨架"))
        self.show_bbox_checkbox.setText(_translate("video_widget", "人物框"))
        self.select_checkbox.setText(_translate("video_widget", "選擇人"))
        self.select_kpt_checkbox.setText(_translate("video_widget", "選擇關節點"))
        self.show_angle_checkbox.setText(_translate("video_widget", "顯示角度"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab), _translate("video_widget", "一般設定"))
        self.id_adjust_groupbox.setTitle(_translate("video_widget", "手動ID修正"))
        self.id_label.setText(_translate("video_widget", "修正ID:"))
        self.label_3.setText(_translate("video_widget", ">"))
        self.id_correct_btn.setText(_translate("video_widget", "修正ID"))
        self.kpt_adjust_groupbox.setTitle(_translate("video_widget", "手動關節點修正"))
        item = self.kpt_table.horizontalHeaderItem(0)
        item.setText(_translate("video_widget", "關節點"))
        item = self.kpt_table.horizontalHeaderItem(1)
        item.setText(_translate("video_widget", "X"))
        item = self.kpt_table.horizontalHeaderItem(2)
        item.setText(_translate("video_widget", "Y"))
        item = self.kpt_table.horizontalHeaderItem(3)
        item.setText(_translate("video_widget", "Z"))
        item = self.kpt_table.horizontalHeaderItem(4)
        item.setText(_translate("video_widget", "有無更改"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab_2), _translate("video_widget", "關節點調整"))
