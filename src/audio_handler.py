import pickle
import pyaudio
import wave
import numpy as np
import pandas as pd

class AudioHandler(object):

    def __init__(self, s, d):
        self.data= d
        self.seconds= s
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.p = None
        self.stream = None
        self.folder = "./audio/"        
        self.bottom = ".wav"
        self.seq_num = -1
        self.output_name= ""      

    def store_data_prediction(self):
        self.p = pyaudio.PyAudio()
#        self.seq_num= pickle.load(open(self.folder_data + "seq_num.pickle", 'rb'))
        self.seq_num += 1
        if (self.seq_num > self.data): 
            self.seq_num= 0
        self.output_name= self.folder + str(self.seq_num) + self.bottom
#        pickle.dump(self.seq_num,open(self.folder_data + "seq_num.pickle", 'wb'))
        
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

