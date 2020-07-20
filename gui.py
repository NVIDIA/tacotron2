import sys
from PyQt5 import Qt
from PyQt5 import QtCore,QtGui
from PyQt5.QtCore import QMutex, QObject, QRunnable, pyqtSignal, pyqtSlot, QThreadPool, QTimer
from PyQt5.QtWidgets import QWidget,QMainWindow,QHeaderView, QMessageBox, QFileDialog
from nvidia_tacotron_TTS_Layout import Ui_MainWindow
from switch import Switch

import time
import requests
import json
import datetime
import numpy as np
import os
import pygame

import matplotlib
import matplotlib.pyplot as plt
import IPython.display as ipd

import sys
sys.path.append(os.path.join(sys.path[0],'waveglow/'))

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser


_mutex1 = QMutex()
_running = False

#https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data
    
    error
        `tuple` (exctype, value, traceback.format_exc() )
    
    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress 

    '''
    
    textready = pyqtSignal(str) 
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    elapsed = pyqtSignal(int)

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()    

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress        
        self.kwargs['elapsed_callback'] = self.signals.elapsed
        self.kwargs['text_ready'] = self.signals.textready

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class GUISignals(QObject):
    progress = pyqtSignal(int)   
    elapsed = pyqtSignal(int)

class GUI(QMainWindow, Ui_MainWindow):
    def __init__(self,app):
        super(GUI, self).__init__()

        self.app = app
        self.setupUi(self)
        self.setWindowTitle("Tacotron2 + Waveglow GUI v%s" %0.1)
        
        self.GpuSwitch = Switch(thumb_radius=8, track_radius=10, show_text = False)
        self.GpuSwitch.setEnabled = torch.cuda.is_available()
        self.use_cuda = False
        self.GpuSwitch.toggled.connect(lambda : self.use_cuda = self.GpuSwitch.isChecked())

        self.model = None
        self.waveglow = None
        self.hparams = None
        
        self.reload_model_flag = True
        self.TTModelCombo.currentIndexChanged.connect(lambda : self.reload_model_flag = True)
        self.WGModelCombo.currentIndexChanged.connect(lambda : self.reload_model_flag = True)
        self.TTSDialogButton.clicked.connect(self.start_synthesis)
        self.TTSSkipButton.clicked.connect(self.skip_wav)
        self.TTSSkipButton.setDisabled(True)
                
        self.logs = []
        self.logs2 = []
        self.max_log_lines = 3
        self.max_log2_lines = 100
        self.TTmodel_dir = [] # Stores list of paths
        self.WGmodel_dir = []
        
        
        # Because of bug in streamelements timestamp filter, need 2 variables for previous time
        
        self.startup_time = datetime.datetime.utcnow().isoformat()
        # self.startup_time = '0' # For debugging
        self.prev_time = datetime.datetime.utcnow().isoformat() 
        #self.prev_time = '0' # for debugging
        self.offset = 0
        
        # self.ClientStartBtn.clicked.connect(self.start_client)
        self.ClientSkipBtn.clicked.connect(self.skip_wav)
        self.channel_id = ''
        # self.ClientStopBtn.clicked.connect(lambda: self.set_client_flag(False))
        self.client_flag = False
        self.ClientStopBtn.setDisabled(True)
        self.ClientSkipBtn.setDisabled(True)
        self.TTModelCombo.setDisabled(True)
        self.WGModelCombo.setDisabled(True)
        self.TTSDialogButton.setDisabled(True)
        self.LoadTTButton.clicked.connect(self.add_TTmodel_path)
        self.LoadWGButton.clicked.connect(self.add_WGmodel_path)
        self.tab_2.setDisabled(True)
        self.log_window2.ensureCursorVisible()
        self.update_log_window("Begin by loading a model")
        pygame.mixer.quit()
        pygame.mixer.init(frequency=22050,size=-16, channels=1)
        self.channel = pygame.mixer.Channel(0)

        self.ClientStartBtn.clicked.connect(self.start)
        self.ClientStopBtn.clicked.connect(self.stop)
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.signals = GUISignals()  
        self.signals.progress.connect(self.update_log_bar)
        self.signals.elapsed.connect(self.print_elapsed)

    
    @pyqtSlot(str)
    def draw_text(self,text):
        # Function to send text from client thread to GUI thread
        # Format of text: <Obj>:<Message>
        obj = text[0:4]
        msg = text[5:]
        if obj=='Log1':
            if len(self.logs) > self.max_log_lines:
                self.logs.pop(0)
            self.logs.append(msg)
            log_text = '\n'.join(self.logs)
            self.log_window1.setText(log_text)
        if obj=='Log2':
            if len(self.logs2) > self.max_log2_lines:
                self.logs2.pop(0)
            self.logs2.append(msg)
            log_text = '\n'.join(self.logs2)
            self.log_window2.setPlainText(log_text)
            self.log_window2.verticalScrollBar().setValue(
                self.log_window2.verticalScrollBar().maximum())
        if obj=='Sta2':
            self.statusbar.setText(msg)
    
    @pyqtSlot(int)
    def update_log_bar(self,val):
        self.progressBar.setValue(val)
        self.progressBar.setTextVisible(val != 0)

    @pyqtSlot(int)
    def update_log_bar2(self,val):
        self.progressBar2.setValue(val)
        self.progressBar2.setTextVisible(val != 0)

    @pyqtSlot(int)
    def print_elapsed(self,val):
        if self.tabWidget.currentIndex()==0:
            self.update_log_window('Elapsed: '+str(val)+'s',mode='overwrite') 
        else:
            pass # No elapsed time for tab2 
    

    def thread_complete(self):
        #print("THREAD COMPLETE!")
        pass

    def print_output(self, s):
        #print(s)
        pass

    def start(self):
        # Pass the function to execute
        global _running
        if not self.validate_se():
            return
        _mutex1.lock()
        _running = True
        _mutex1.unlock()
        worker = Worker(self.execute_this_fn, self.channel) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.update_log_bar2)
        worker.signals.textready.connect(self.draw_text)
        worker.signals.elapsed.connect(self.print_elapsed)
        
        # Execute
        self.threadpool.start(worker) 
        
    def stop(self):
        global _running
        _mutex1.lock()
        _running = False
        _mutex1.unlock()
        self.skip_wav()

    # def progress_fn(self, n):
    #     print("%d%% done" % n)
    
    def execute_this_fn(self, channel, progress_callback, elapsed_callback, text_ready):
        # Function executes in client thread. 
        # Synthesis does not block gui thread.
        # Can run methods of GUI class but cannot run GUI updates directly
        # from this thread. Use signal and slots to communicate with main GUI.
        # We pass pygame.mixer.channel object into this thread to check for channel activity.
        self.ClientStartBtn.setDisabled(True)
        self.ClientStopBtn.setEnabled(True)
        self.ClientSkipBtn.setEnabled(True)
        self.tab.setDisabled(True)
        self.ClientAmountLine.setDisabled(True)
        text_ready.emit("Sta2:Connecting to StreamElements")
        url = "https://api.streamelements.com/kappa/v2/tips/"+self.channel_id
        headers = {'accept': 'application/json',"Authorization": "Bearer "+self.get_token()}
        text_ready.emit('Log2:Using TTS Voice: '+self.model_selected)
        text_ready.emit('Log2:Initializing')
        min_donation = self.get_min_donation()
        text_ready.emit('Log2:Minimum amount for TTS: '+str(min_donation))
        graph = Graph(mode="synthesize")
        model_fpath = self.get_current_model_dir()
        while True:
            _mutex1.lock()
            if _running == False:
                _mutex1.unlock()
                break
            else:
                _mutex1.unlock()
            if not channel.get_busy():
                #print('Polling', datetime.datetime.utcnow().isoformat())
                text_ready.emit("Sta2:Waiting for incoming donations . . .")
                current_time = datetime.datetime.utcnow().isoformat()
                querystring = {"offset":self.offset,"limit":"1","sort":"createdAt","after":self.startup_time,"before":current_time}
                response = requests.request("GET", url, headers=headers, params=querystring)
                data = json.loads(response.text)
                for dono in data['docs']:
                    text_ready.emit("Sta2:Processing donations")
                    dono_time = dono['createdAt']
                    self.offset += 1
                    if dono_time > self.prev_time: # Str comparison
                        amount = dono['donation']['amount'] # Int
                        if float(amount) >= min_donation: # Float comparison
                            name = dono['donation']['user']['username']
                            msg = dono['donation']['message']
                            ## TODO Allow multiple speaker in msg
                            msg = self.pre_process_str(msg)
                            currency = dono['donation']['currency']
                            dono_id = dono['_id']
                            text_ready.emit("Log2:\n###########################")
                            text_ready.emit("Log2:"+name+' donated '+currency+str(amount))
                            text_ready.emit("Log2:"+msg)
                            wav=synthesize.synthesize(msg, model_fpath, graph, progress_callback, elapsed_callback)
                            self.playback_wav(wav)
                            self.prev_time = dono_time # Increment time
            time.sleep(0.5)
        self.ClientStartBtn.setEnabled(True)
        self.ClientStopBtn.setDisabled(True)
        self.ClientSkipBtn.setDisabled(True)
        self.tab.setEnabled(True)
        self.ClientAmountLine.setEnabled(True)
        text_ready.emit('Log2:\nDisconnected')
        text_ready.emit('Sta2:Ready')
        return 'Return value of execute_this_fn'    


    def startup_update(self):
        if not self.tab_2.isEnabled():
            self.tab_2.setEnabled(True)
        if not self.TTSDialogButton.isEnabled():
            self.TTSDialogButton.setEnabled(True)

    def add_TTmodel_path(self):
        fpath = str(QFileDialog.getOpenFileName(self, 
                                            'Select Tacotron2 model', 
                                            filter='*.pt')[0])
        print(fpath)
        if not fpath: # If no folder selected
            return
        if fpath not in self.TTmodel_dir:
            head,tail = os.path.split(fpath) # Split into parent and child dir
            self.TTmodel_dir.append(fpath) # Save full path
            self.populate_modelcombo(tail, self.TTModelCombo)
            self.update_log_window("Added Tacotron 2 model: "+tail)
            if self.WGModelCombo.count() > 0:
                self.startup_update()

    def add_WGmodel_path(self):
        fpath = str(QFileDialog.getOpenFileName(self, 
                                            'Select Waveglow model', 
                                            filter='*.pt')[0])
        if not fpath: # If no folder selected
            return
        if fpath not in self.WGmodel_dir:
            head,tail = os.path.split(fpath) # Split into parent and child dir
            self.WGmodel_dir.append(fpath) # Save full path
            self.populate_modelcombo(tail, self.WGModelCombo)
            self.update_log_window("Added Waveglow model: "+tail)
            if self.TTModelCombo.count() > 0:
                self.startup_update()

    def populate_modelcombo(self, item, combobox):
        combobox.addItem(item)
        combobox.setCurrentIndex(combobox.count()-1)
        if not combobox.isEnabled():
            combobox.setEnabled(True)  

    def get_current_TTmodel_dir(self):
        return self.TTmodel_dir[self.TTModelCombo.currentIndex()]

    def get_current_WGmodel_dir(self):
        return self.WGmodel_dir[self.WGModelCombo.currentIndex()]

    def update_log_window(self, line, mode="newline"):
        if mode == "newline" or not self.logs:
            self.logs.append(line)
            if len(self.logs) > self.max_log_lines:
                del self.logs[0]
        elif mode == "append":
            self.logs[-1] += line
        elif mode == "overwrite":
            self.logs[-1] = line
        log_text = '\n'.join(self.logs)
        
        self.log_window1.setText(log_text)
        self.app.processEvents()

    def playback_wav(self,wav):
        if self.tabWidget.currentIndex()==0:
            self.TTSSkipButton.setEnabled(True)
        else:
            self.ClientSkipBtn.setEnabled(True)
        sound = pygame.mixer.Sound(wav)
        self.channel.queue(sound)
        # TODO Disable skip btn on playback end

    def skip_wav(self):
        if self.channel.get_busy():
            self.channel.stop()
        self.TTSSkipButton.setDisabled(True)
        self.ClientSkipBtn.setDisabled(True)

    def reload_model(self):
        TTmodel_fpath = self.get_current_TTmodel_dir()
        WGmodel_fpath = self.get_current_WGmodel_dir()
        # Setup hparams
        self.hparams = create_hparams()
        self.hparams.sampling_rate = 22050
        # Load Tacotron 2 from checkpoint
        self.model = load_model(self.hparams,self.use_cuda)
        self.model.load_state_dict(torch.load(TTmodel_fpath)['state_dict'])
        if self.use_cuda:
            _ = self.model.cuda().eval().half()
        else:
            _ = self.model.eval()
        #  Load WaveGlow for mel2audio synthesis and denoiser
        self.waveglow = torch.load(WGmodel_fpath)['model']
        self.waveglow.use_cuda = self.use_cuda
        if self.use_cuda: 
            self.waveglow.cuda().eval().half()
        else: 
            self.waveglow.eval()
        for k in self.waveglow.convinv:
            k.float()
        #denoiser = Denoiser(waveglow,use_cuda=self.use_cuda)

    def start_synthesis(self):
        # Runs in main gui thread. Synthesize blocks gui.
        # Can update gui directly in this function.
        text = self.TTSTextEdit.toPlainText()
        text = self.pre_process_str(text)
        self.TTSDialogButton.setDisabled(True)
        self.TTModelCombo.setDisabled(True)
        self.WGModelCombo.setDisabled(True)
        self.TTSTextEdit.setDisabled(True)
        self.LoadTTButton.setDisabled(True)
        self.update_log_bar(0)
        self.update_log_window('Initializing')
        self.update_status_bar("Creating voice")
        # We use a signal callback here to stick to the same params type in synthesize.py
        if self.reload_model_flag: 
            self.reload_model()
        self.playback_wav(wav)
        self.update_log_window('Done')
        self.TTSDialogButton.setEnabled(True)
        self.TTModelCombo.setEnabled(True)
        self.WGModelCombo.setEnabled(True)
        self.TTSTextEdit.setEnabled(True)
        self.LoadTTButton.setEnabled(True)
        self.update_status_bar("Ready")
        # TODO get pygame mixer callback on end or use sounddevice
        
    def update_log_window_2(self, line, mode="newline"):
        if mode == "newline" or not self.logs2:
            self.logs2.append(line)
        elif mode == "append":
            self.logs2[-1] += line
        elif mode == "overwrite":
            self.logs2[-1] = line
        log_text = '\n'.join(self.logs2)
        
        self.log_window2.setPlainText(log_text)
        self.log_window2.verticalScrollBar().setValue(
            self.log_window2.verticalScrollBar().maximum())
        self.app.processEvents()

    def update_status_bar(self, line):
        self.statusbar.setText(line)
        self.app.processEvents()


    def get_token(self):
        TOKEN = ''.join(self.APIKeyLine.text().split())
        return TOKEN

    def set_client_flag(self,val):
        self.client_flag = val
   
    def validate_se(self):
        # Connect to streamelement and saves channel id
        # return true if chn id and token returns valid
        # Test Channel ID
        self.update_status_bar("Validating StreamElements")
        CHANNEL_NAME = ''.join(self.ChannelName.text().split())
        url = "https://api.streamelements.com/kappa/v2/channels/"+CHANNEL_NAME
        response = requests.request("GET", url, headers={'accept': 'application/json'})
        if response.status_code == 200:
            # Test JWT Token
            self.channel_id = json.loads(response.text)['_id']
            url = "https://api.streamelements.com/kappa/v2/tips/"+self.channel_id
            querystring = {"offset":"0","limit":"10","sort":"createdAt","after":"0","before":"0"}
            TOKEN = self.get_token()
            headers = {'accept': 'application/json',"Authorization": "Bearer "+TOKEN}
            response2 = requests.request("GET", url, headers=headers, params=querystring)
            if response2.status_code == 200:
                self.update_log_window_2("\nConnected to "+CHANNEL_NAME)
                self.set_client_flag(True)
                return True
            else: 
                self.update_log_window_2("\nDouble check your token")
                print(response2.text)
        else: 
            self.update_log_window_2("\nDouble check your channel name")
            print(response.text)
        
        return False

    def pre_process_str(self,msg):
        # Replaces digits with words
        msg = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), msg)
        return msg
    
    def get_min_donation(self):
        return float(self.ClientAmountLine.value())

  

if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = GUI(app)
    window.show()
    sys.exit(app.exec_())