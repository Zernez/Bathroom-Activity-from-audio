import pickle
import pyaudio
import wave
import numpy as np
import pandas as pd
from math import log10
import audioop
import time

class AudioHandler(object):

    def __init__(self, s, d):
        self.data= d
        self.seconds= s
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.rms = 0
        self.p = None
        self.stream = None
        self.folder = "./audio/"        
        self.bottom = ".wav"
        self.seq_num = -1
        self.output_name= ""      

    def store_data_prediction(self):
        self.p = pyaudio.PyAudio()
        self.seq_num += 1
        if (self.seq_num > self.data): 
            self.seq_num= 0
        self.output_name= self.folder + str(self.seq_num) + self.bottom
        
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  frames_per_buffer=self.CHUNK)
        
        frames= []
        for i in range (0, int (self.RATE/self.CHUNK *self.seconds)):
            data= self.stream.read(self.CHUNK)
            frames.append(data)
                 
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
        wf= wave.open(self.output_name, "wb")
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes (b''.join(frames))
        wf.close()
            
        return self.seq_num

    def check_level_sound(self):
        self.p = pyaudio.PyAudio()
        rms= 0
        
        def callback(in_data, frame_count, time_info, status):
            self.rms = audioop.rms(in_data, 2) / 32767
            return in_data, pyaudio.paContinue        

        
        self.stream = self.p.open(format=self.FORMAT,
                                  channels= 1,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback= callback)
        
        self.stream.start_stream()
        
        dB_level= []
        i= 0
        for i in range (0, 4):
            if (self.rms== 0):
                self.rms= 0.001
            dB_level.append(20 * log10(self.rms))
            i+= 1
            time.sleep(0.2)
                 
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
        return np.max(dB_level)