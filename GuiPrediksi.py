from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.uic.properties import QtCore

from main import predict

from v3r import Ui_MainWindow
import sys

class window(QtWidgets.QMainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.input_list_lineInput = []
        self.input_list = []
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('./icongui.png'))
        self.verticalLayoutWidget = self.ui.verticalLayoutWidget
        self.verticalLayoutWidget2 = self.ui.verticalLayoutWidget_2
        self.ui.addButton.clicked.connect(self.add_input_box)
        self.ui.deleteButton.clicked.connect(self.onDelete)
        self.ui.checkButton.clicked.connect(self.loaddata)
        self.ui.actionreset.triggered.connect(self.reset)
        self.ui.actionquit.triggered.connect(self.close)


    def add_input_box(self):
        if len(self.input_list_lineInput) > 6:
            print("Limit boss")
            return
        self.lineInput = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineInput.setObjectName("lineInput")
        self.input_list_lineInput.append(self.lineInput)
        self.ui.inputNameLayout.addWidget(self.lineInput)


    def reset(self):
        self.delete = self.ui.inputNameLayout
        while self.delete.count() > 0:
            removed = self.delete.itemAt(self.delete.count() - 1).widget()
            self.delete.removeWidget(removed)
            removed.setParent(None)
            removed.deleteLater()
        self.ui.resultTable.setRowCount(0)
        self.input_list_lineInput = []
        self.input_list = []

    def close(self):
        sys.exit(   )

    def onDelete(self):
        self.delete = self.ui.inputNameLayout
        removed = self.delete.itemAt(self.delete.count() - 1).widget()
        self.delete.removeWidget(removed)
        removed.setParent(None)
        removed.deleteLater()
        self.input_list_lineInput.pop(-1)
        if(self.input_list_lineInput == []):
            self.input_list.pop(-1)

    def loaddata(self):
        for item in self.input_list_lineInput:
            self.input_list.append(item.text())

        pred = predict(self.input_list).to_numpy()

        self.ui.resultTable.setRowCount(len(pred))
        row = 0

        for person in pred:
            self.ui.resultTable.setItem(row, 0, QtWidgets.QTableWidgetItem(person[0]))
            self.ui.resultTable.setItem(row, 1, QtWidgets.QTableWidgetItem(person[1]))
            self.ui.resultTable.setItem(row, 2, QtWidgets.QTableWidgetItem(str("{:.2f}".format(person[2]))))
            row = row + 1



def create_app():
    app = QtWidgets.QApplication(sys.argv)
    win = window()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    create_app()

