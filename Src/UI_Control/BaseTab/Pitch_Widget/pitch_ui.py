# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\user\Desktop\Pitch-Skeleton-User-Interface\Src\UI_Control\BaseTab\Pitch_Widget\pitch.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Pitch_UI(object):
    def setupUi(self, Pitch_UI):
        Pitch_UI.setObjectName("Pitch_UI")
        Pitch_UI.resize(1200, 800)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(Pitch_UI)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.ResolutionLabel = QtWidgets.QLabel(Pitch_UI)
        self.ResolutionLabel.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.ResolutionLabel.setFont(font)
        self.ResolutionLabel.setObjectName("ResolutionLabel")
        self.verticalLayout_2.addWidget(self.ResolutionLabel)
        self.FrameView = QtWidgets.QGraphicsView(Pitch_UI)
        self.FrameView.setMinimumSize(QtCore.QSize(480, 360))
        self.FrameView.setObjectName("FrameView")
        self.verticalLayout_2.addWidget(self.FrameView)
        self.CurveView = QtWidgets.QGraphicsView(Pitch_UI)
        self.CurveView.setObjectName("CurveView")
        self.verticalLayout_2.addWidget(self.CurveView)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.backKeyBtn = QtWidgets.QPushButton(Pitch_UI)
        self.backKeyBtn.setMinimumSize(QtCore.QSize(50, 30))
        self.backKeyBtn.setMaximumSize(QtCore.QSize(50, 30))
        self.backKeyBtn.setObjectName("backKeyBtn")
        self.horizontalLayout_4.addWidget(self.backKeyBtn)
        self.playBtn = QtWidgets.QPushButton(Pitch_UI)
        self.playBtn.setMinimumSize(QtCore.QSize(50, 30))
        self.playBtn.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.playBtn.setFont(font)
        self.playBtn.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.playBtn.setIconSize(QtCore.QSize(20, 20))
        self.playBtn.setObjectName("playBtn")
        self.horizontalLayout_4.addWidget(self.playBtn)
        self.forwardKeyBtn = QtWidgets.QPushButton(Pitch_UI)
        self.forwardKeyBtn.setMinimumSize(QtCore.QSize(50, 30))
        self.forwardKeyBtn.setMaximumSize(QtCore.QSize(50, 30))
        self.forwardKeyBtn.setObjectName("forwardKeyBtn")
        self.horizontalLayout_4.addWidget(self.forwardKeyBtn)
        self.frameSlider = QtWidgets.QSlider(Pitch_UI)
        self.frameSlider.setMinimumSize(QtCore.QSize(300, 30))
        self.frameSlider.setOrientation(QtCore.Qt.Horizontal)
        self.frameSlider.setObjectName("frameSlider")
        self.horizontalLayout_4.addWidget(self.frameSlider)
        self.frameNumLabel = QtWidgets.QLabel(Pitch_UI)
        self.frameNumLabel.setMinimumSize(QtCore.QSize(20, 20))
        self.frameNumLabel.setObjectName("frameNumLabel")
        self.horizontalLayout_4.addWidget(self.frameNumLabel)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout_2.setStretch(1, 3)
        self.verticalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.tabWidget = QtWidgets.QTabWidget(Pitch_UI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMaximumSize(QtCore.QSize(500, 16777215))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.setting_groupbox = QtWidgets.QGroupBox(self.tab)
        self.setting_groupbox.setMinimumSize(QtCore.QSize(450, 625))
        self.setting_groupbox.setMaximumSize(QtCore.QSize(450, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.setting_groupbox.setFont(font)
        self.setting_groupbox.setObjectName("setting_groupbox")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.setting_groupbox)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.camerSettingGroupBox = QtWidgets.QGroupBox(self.setting_groupbox)
        self.camerSettingGroupBox.setObjectName("camerSettingGroupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.camerSettingGroupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.cameraCheckBox = QtWidgets.QCheckBox(self.camerSettingGroupBox)
        self.cameraCheckBox.setObjectName("cameraCheckBox")
        self.gridLayout.addWidget(self.cameraCheckBox, 4, 0, 1, 1)
        self.recordCheckBox = QtWidgets.QCheckBox(self.camerSettingGroupBox)
        self.recordCheckBox.setObjectName("recordCheckBox")
        self.gridLayout.addWidget(self.recordCheckBox, 4, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cameraIdLabel = QtWidgets.QLabel(self.camerSettingGroupBox)
        self.cameraIdLabel.setObjectName("cameraIdLabel")
        self.horizontalLayout.addWidget(self.cameraIdLabel)
        self.cameraIdInput = QtWidgets.QSpinBox(self.camerSettingGroupBox)
        self.cameraIdInput.setMinimumSize(QtCore.QSize(0, 0))
        self.cameraIdInput.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.cameraIdInput.setObjectName("cameraIdInput")
        self.horizontalLayout.addWidget(self.cameraIdInput)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.startPitchCheckBox = QtWidgets.QCheckBox(self.camerSettingGroupBox)
        self.startPitchCheckBox.setObjectName("startPitchCheckBox")
        self.gridLayout.addWidget(self.startPitchCheckBox, 4, 2, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.FPSLabel = QtWidgets.QLabel(self.camerSettingGroupBox)
        self.FPSLabel.setObjectName("FPSLabel")
        self.horizontalLayout_3.addWidget(self.FPSLabel)
        self.FPSInfoLabel = QtWidgets.QLabel(self.camerSettingGroupBox)
        self.FPSInfoLabel.setMinimumSize(QtCore.QSize(130, 0))
        self.FPSInfoLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.FPSInfoLabel.setObjectName("FPSInfoLabel")
        self.horizontalLayout_3.addWidget(self.FPSInfoLabel)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        self.verticalLayout_7.addWidget(self.camerSettingGroupBox)
        self.displayGroupBox = QtWidgets.QGroupBox(self.setting_groupbox)
        self.displayGroupBox.setObjectName("displayGroupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.displayGroupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.showSkeletonCheckBox = QtWidgets.QCheckBox(self.displayGroupBox)
        self.showSkeletonCheckBox.setChecked(False)
        self.showSkeletonCheckBox.setObjectName("showSkeletonCheckBox")
        self.gridLayout_2.addWidget(self.showSkeletonCheckBox, 1, 0, 1, 1)
        self.showBboxCheckBox = QtWidgets.QCheckBox(self.displayGroupBox)
        self.showBboxCheckBox.setCheckable(True)
        self.showBboxCheckBox.setChecked(False)
        self.showBboxCheckBox.setObjectName("showBboxCheckBox")
        self.gridLayout_2.addWidget(self.showBboxCheckBox, 1, 1, 1, 1)
        self.selectKptCheckBox = QtWidgets.QCheckBox(self.displayGroupBox)
        self.selectKptCheckBox.setObjectName("selectKptCheckBox")
        self.gridLayout_2.addWidget(self.selectKptCheckBox, 3, 0, 1, 1)
        self.showAngleCheckBox = QtWidgets.QCheckBox(self.displayGroupBox)
        self.showAngleCheckBox.setObjectName("showAngleCheckBox")
        self.gridLayout_2.addWidget(self.showAngleCheckBox, 3, 1, 1, 1)
        self.selectCheckBox = QtWidgets.QCheckBox(self.displayGroupBox)
        self.selectCheckBox.setChecked(False)
        self.selectCheckBox.setObjectName("selectCheckBox")
        self.gridLayout_2.addWidget(self.selectCheckBox, 2, 1, 1, 1)
        self.showLineCheckBox = QtWidgets.QCheckBox(self.displayGroupBox)
        self.showLineCheckBox.setObjectName("showLineCheckBox")
        self.gridLayout_2.addWidget(self.showLineCheckBox, 2, 0, 1, 1)
        self.pitchLabel = QtWidgets.QLabel(self.displayGroupBox)
        self.pitchLabel.setObjectName("pitchLabel")
        self.gridLayout_2.addWidget(self.pitchLabel, 0, 0, 1, 1)
        self.pitchInput = QtWidgets.QComboBox(self.displayGroupBox)
        self.pitchInput.setObjectName("pitchInput")
        self.pitchInput.addItem("")
        self.pitchInput.addItem("")
        self.gridLayout_2.addWidget(self.pitchInput, 0, 1, 1, 1)
        self.verticalLayout_7.addWidget(self.displayGroupBox)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_7.addItem(spacerItem)
        self.displayGroupBox.raise_()
        self.camerSettingGroupBox.raise_()
        self.verticalLayout.addWidget(self.setting_groupbox)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.kptAdjustGroupbox = QtWidgets.QGroupBox(self.tab_2)
        self.kptAdjustGroupbox.setObjectName("kptAdjustGroupbox")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.kptAdjustGroupbox)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.KptTable = QtWidgets.QTableWidget(self.kptAdjustGroupbox)
        self.KptTable.setMinimumSize(QtCore.QSize(400, 0))
        self.KptTable.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.KptTable.setObjectName("KptTable")
        self.KptTable.setColumnCount(4)
        self.KptTable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.KptTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.KptTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.KptTable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.KptTable.setHorizontalHeaderItem(3, item)
        self.verticalLayout_6.addWidget(self.KptTable)
        self.verticalLayout_3.addWidget(self.kptAdjustGroupbox)
        self.tabWidget.addTab(self.tab_2, "")
        self.horizontalLayout_2.addWidget(self.tabWidget)

        self.retranslateUi(Pitch_UI)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Pitch_UI)

    def retranslateUi(self, Pitch_UI):
        _translate = QtCore.QCoreApplication.translate
        Pitch_UI.setWindowTitle(_translate("Pitch_UI", "Form"))
        self.ResolutionLabel.setText(_translate("Pitch_UI", "(0, 0) - "))
        self.backKeyBtn.setText(_translate("Pitch_UI", "<<"))
        self.playBtn.setText(_translate("Pitch_UI", "▶︎"))
        self.forwardKeyBtn.setText(_translate("Pitch_UI", ">>"))
        self.frameNumLabel.setText(_translate("Pitch_UI", "0/0"))
        self.setting_groupbox.setTitle(_translate("Pitch_UI", "2D 關節點"))
        self.camerSettingGroupBox.setTitle(_translate("Pitch_UI", "相機設定"))
        self.cameraCheckBox.setText(_translate("Pitch_UI", "開啟相機"))
        self.recordCheckBox.setText(_translate("Pitch_UI", "開始錄影"))
        self.cameraIdLabel.setText(_translate("Pitch_UI", "相機ID:"))
        self.startPitchCheckBox.setText(_translate("Pitch_UI", "開始投球"))
        self.FPSLabel.setText(_translate("Pitch_UI", "FPS:"))
        self.FPSInfoLabel.setText(_translate("Pitch_UI", "0"))
        self.displayGroupBox.setTitle(_translate("Pitch_UI", "顯示資訊"))
        self.showSkeletonCheckBox.setText(_translate("Pitch_UI", "人體骨架"))
        self.showBboxCheckBox.setText(_translate("Pitch_UI", "人物框"))
        self.selectKptCheckBox.setText(_translate("Pitch_UI", "選擇關節點"))
        self.showAngleCheckBox.setText(_translate("Pitch_UI", "顯示角度"))
        self.selectCheckBox.setText(_translate("Pitch_UI", "選擇人"))
        self.showLineCheckBox.setText(_translate("Pitch_UI", "輔助線"))
        self.pitchLabel.setText(_translate("Pitch_UI", "投手:"))
        self.pitchInput.setItemText(0, _translate("Pitch_UI", "右投"))
        self.pitchInput.setItemText(1, _translate("Pitch_UI", "左投"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Pitch_UI", "一般設定"))
        self.kptAdjustGroupbox.setTitle(_translate("Pitch_UI", "手動關節點修正"))
        item = self.KptTable.horizontalHeaderItem(0)
        item.setText(_translate("Pitch_UI", "Keypoint"))
        item = self.KptTable.horizontalHeaderItem(1)
        item.setText(_translate("Pitch_UI", "X"))
        item = self.KptTable.horizontalHeaderItem(2)
        item.setText(_translate("Pitch_UI", "Y"))
        item = self.KptTable.horizontalHeaderItem(3)
        item.setText(_translate("Pitch_UI", "有無更改"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Pitch_UI", "關節點調整"))