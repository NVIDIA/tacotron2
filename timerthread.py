# from https://stackoverflow.com/a/14369192
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread
import time

class timerThread(QThread):
    timeElapsed = pyqtSignal(int)

    def __init__(self, timeoffset, parent=None):
        super(timerThread, self).__init__(parent)
        self.timeoffset = timeoffset
        self.timeStart = None
    

    def start(self, timeStart):
        self.timeStart = timeStart

        return super(timerThread, self).start()

    def run(self):
        while self.parent().isRunning():
            self.timeElapsed.emit(time.time() - self.timeStart + self.timeoffset)
            time.sleep(1)
   