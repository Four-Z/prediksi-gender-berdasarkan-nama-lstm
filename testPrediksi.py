# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testPrediksi.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(490, 406)
        MainWindow.setFixedSize(490, 406)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.namebox = QtWidgets.QGroupBox(self.centralwidget)
        self.namebox.setGeometry(QtCore.QRect(40, 50, 191, 121))
        self.namebox.setObjectName("namebox")
        self.nametext = QtWidgets.QLineEdit(self.namebox)
        self.nametext.setGeometry(QtCore.QRect(30, 40, 113, 20))
        self.nametext.setObjectName("nametext")
        self.addbutton = QtWidgets.QPushButton(self.centralwidget)
        self.addbutton.setGeometry(QtCore.QRect(30, 210, 75, 23))
        self.addbutton.setObjectName("addbutton")
        self.deletebutton = QtWidgets.QPushButton(self.centralwidget)
        self.deletebutton.setGeometry(QtCore.QRect(140, 210, 75, 23))
        self.deletebutton.setObjectName("deletebutton")
        self.checkbutton = QtWidgets.QPushButton(self.centralwidget)
        self.checkbutton.setGeometry(QtCore.QRect(80, 250, 75, 23))
        self.checkbutton.setObjectName("checkbutton")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(270, 70, 211, 192))
        self.tableView.setObjectName("tableView")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.namebox.setTitle(_translate("MainWindow", "Masukkan Nama :"))
        self.addbutton.setText(_translate("MainWindow", "Add"))
        self.deletebutton.setText(_translate("MainWindow", "Delete"))
        self.checkbutton.setText(_translate("MainWindow", "Check"))
