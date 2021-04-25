from PyQt5 import QtCore, QtGui, QtWidgets
from functions.backgroundsubtraction import BackgroundSubtraction
from functions.opticalflow import OpticalFlow
from functions.perspectivetransform import PerspectiveTransform
from functions.pca import PCA

# open -a Designer
# pyuic5 -o ui.py ui.ui


class Ui_MainWindow(object):
    def __init__(self):
        self.BackgroundSubtraction = BackgroundSubtraction()
        self.OpticalFlow = OpticalFlow()
        self.PerspectiveTransform = PerspectiveTransform()
        self.PCA = PCA()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(316, 469)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 50, 281, 31))
        self.pushButton.setObjectName("1.1 Backgournd Subtraction")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 140, 281, 31))
        self.pushButton_2.setObjectName("2.1 Preprocessing")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 170, 281, 31))
        self.pushButton_3.setObjectName("2.2 Video tracking")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(20, 270, 281, 31))
        self.pushButton_4.setObjectName("3.1 Perspective Transform")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 370, 281, 31))
        self.pushButton_5.setObjectName("4.1 Image Reconstruction")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 400, 281, 31))
        self.pushButton_6.setObjectName("4.2 Compute the Reconstruction Error")

        font = QtGui.QFont()
        font.setPointSize(18)


        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 20, 300, 16))
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 110, 300, 16))
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 240, 300, 16))
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 340, 300, 16))
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.BackgroundSubtraction.subtract)
        self.pushButton_2.clicked.connect(self.OpticalFlow.draw)
        self.pushButton_3.clicked.connect(self.OpticalFlow.flow)
        self.pushButton_4.clicked.connect(self.PerspectiveTransform.transform)
        self.pushButton_5.clicked.connect(self.PCA.reconstruction)
        self.pushButton_6.clicked.connect(self.PCA.get_error)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "1.1 Backgournd Subtraction"))
        self.pushButton_2.setText(_translate("MainWindow", "2.1 Preprocessing"))
        self.pushButton_3.setText(_translate("MainWindow", "2.2 Video tracking"))
        self.pushButton_4.setText(_translate("MainWindow", "3.1 Perspective Transform"))
        self.pushButton_5.setText(_translate("MainWindow", "4.1 Image Reconstruction"))
        self.pushButton_6.setText(_translate("MainWindow", "4.2 Compute the Reconstruction Error"))
        self.label.setText(_translate("MainWindow", "1.  Background Subtraction"))
        self.label_2.setText(_translate("MainWindow", "2.  Optical Flow"))
        self.label_3.setText(_translate("MainWindow", "3.  Perspective Transform"))
        self.label_4.setText(_translate("MainWindow", "4.  PCA"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
