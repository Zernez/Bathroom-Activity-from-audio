import pyaudio
import wave
import numpy as np

data= 5
seconds= 5
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
p = None
stream = None
folder = ""        
bottom = ".wav"
seq_num = -1
output_name= ""      


p = pyaudio.PyAudio()

seq_num += 1

if (seq_num > data): 
    seq_num= 0
    
output_name= folder + str(seq_num) + bottom
        
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                frames_per_buffer=CHUNK)
        
frames= []
for i in range (0, int (RATE/CHUNK *seconds)):
    data= stream.read(CHUNK)
    frames.append(data)
                 
stream.stop_stream()
stream.close()
p.terminate()
        
wf= wave.open(folder + output_name, "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes (b''.join(frames))
wf.close()