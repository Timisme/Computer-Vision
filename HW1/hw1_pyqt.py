# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw1cv.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(997, 663)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.calibration = QtWidgets.QLabel(self.centralwidget)
        self.calibration.setGeometry(QtCore.QRect(40, 130, 141, 81))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.calibration.setFont(font)
        self.calibration.setScaledContents(False)
        self.calibration.setObjectName("calibration")
        self.find_intrin = QtWidgets.QPushButton(self.centralwidget)
        self.find_intrin.setGeometry(QtCore.QRect(70, 320, 191, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.find_intrin.setFont(font)
        self.find_intrin.setObjectName("find_intrin")
        self.find_distortion = QtWidgets.QPushButton(self.centralwidget)
        self.find_distortion.setGeometry(QtCore.QRect(70, 410, 191, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.find_distortion.setFont(font)
        self.find_distortion.setObjectName("find_distortion")
        self.find_corners = QtWidgets.QPushButton(self.centralwidget)
        self.find_corners.setGeometry(QtCore.QRect(70, 240, 191, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.find_corners.setFont(font)
        self.find_corners.setObjectName("find_corners")
        self.find_extrinsic = QtWidgets.QLabel(self.centralwidget)
        self.find_extrinsic.setGeometry(QtCore.QRect(320, 190, 171, 81))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.find_extrinsic.setFont(font)
        self.find_extrinsic.setScaledContents(False)
        self.find_extrinsic.setObjectName("find_extrinsic")
        self.select_img = QtWidgets.QLabel(self.centralwidget)
        self.select_img.setGeometry(QtCore.QRect(350, 260, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.select_img.setFont(font)
        self.select_img.setScaledContents(False)
        self.select_img.setObjectName("select_img")
        self.find_extrin = QtWidgets.QPushButton(self.centralwidget)
        self.find_extrin.setGeometry(QtCore.QRect(320, 390, 191, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.find_extrin.setFont(font)
        self.find_extrin.setObjectName("find_extrin")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(350, 310, 131, 41))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.augmented = QtWidgets.QLabel(self.centralwidget)
        self.augmented.setGeometry(QtCore.QRect(590, 30, 281, 81))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.augmented.setFont(font)
        self.augmented.setScaledContents(False)
        self.augmented.setObjectName("augmented")
        self.aug_show = QtWidgets.QPushButton(self.centralwidget)
        self.aug_show.setGeometry(QtCore.QRect(590, 110, 191, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.aug_show.setFont(font)
        self.aug_show.setObjectName("aug_show")
        self.disparity = QtWidgets.QLabel(self.centralwidget)
        self.disparity.setGeometry(QtCore.QRect(590, 200, 281, 81))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.disparity.setFont(font)
        self.disparity.setScaledContents(False)
        self.disparity.setObjectName("disparity")
        self.disparity_show = QtWidgets.QPushButton(self.centralwidget)
        self.disparity_show.setGeometry(QtCore.QRect(590, 270, 191, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.disparity_show.setFont(font)
        self.disparity_show.setObjectName("disparity_show")
        self.sift = QtWidgets.QLabel(self.centralwidget)
        self.sift.setGeometry(QtCore.QRect(590, 370, 281, 81))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.sift.setFont(font)
        self.sift.setScaledContents(False)
        self.sift.setObjectName("sift")
        self.sift_keypnt = QtWidgets.QPushButton(self.centralwidget)
        self.sift_keypnt.setGeometry(QtCore.QRect(590, 430, 331, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.sift_keypnt.setFont(font)
        self.sift_keypnt.setObjectName("sift_keypnt")
        self.sift_matched = QtWidgets.QPushButton(self.centralwidget)
        self.sift_matched.setGeometry(QtCore.QRect(590, 500, 331, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.sift_matched.setFont(font)
        self.sift_matched.setObjectName("sift_matched")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 997, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.calibration.setText(_translate("MainWindow", "1. Calibration"))
        self.find_intrin.setText(_translate("MainWindow", "1.2 Find Intrinsic"))
        self.find_distortion.setText(_translate("MainWindow", "1.4 Find Distortion"))
        self.find_corners.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.find_extrinsic.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.select_img.setText(_translate("MainWindow", "Select Image"))
        self.find_extrin.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.comboBox.setItemText(0, _translate("MainWindow", "1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "2"))
        self.comboBox.setItemText(2, _translate("MainWindow", "3"))
        self.comboBox.setItemText(3, _translate("MainWindow", "4"))
        self.comboBox.setItemText(4, _translate("MainWindow", "5"))
        self.comboBox.setItemText(5, _translate("MainWindow", "6"))
        self.comboBox.setItemText(6, _translate("MainWindow", "7"))
        self.comboBox.setItemText(7, _translate("MainWindow", "8"))
        self.comboBox.setItemText(8, _translate("MainWindow", "9"))
        self.comboBox.setItemText(9, _translate("MainWindow", "10"))
        self.comboBox.setItemText(10, _translate("MainWindow", "11"))
        self.comboBox.setItemText(11, _translate("MainWindow", "12"))
        self.comboBox.setItemText(12, _translate("MainWindow", "13"))
        self.comboBox.setItemText(13, _translate("MainWindow", "14"))
        self.comboBox.setItemText(14, _translate("MainWindow", "15"))
        self.augmented.setText(_translate("MainWindow", "2. Augmented Reality"))
        self.aug_show.setText(_translate("MainWindow", "2. Show images"))
        self.disparity.setText(_translate("MainWindow", "3. Disparity Map"))
        self.disparity_show.setText(_translate("MainWindow", "3. Show image"))
        self.sift.setText(_translate("MainWindow", "4. SIFT"))
        self.sift_keypnt.setText(_translate("MainWindow", "4.1 Show Keypoints"))
        self.sift_matched.setText(_translate("MainWindow", "4.2 Show Matched Keypoins"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
