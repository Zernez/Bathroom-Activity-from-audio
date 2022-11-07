from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import time
from joblib import dump, load
import time
import os
import os.path
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import alexnet


class PredictorAlx:

    def __init__(self):  
        self.activities= ['toilet_flush', 'showering', 'vacuum_cleaner', 'brushing_teeth', 'washing_machine'] 
        self.img_x= 128
        self.img_y= 431        
        self.alexnet_model = alexnet()
        self.learning_rate = 2e-4
        self.optimizer = optim.Adam(self.alexnet_model.parameters(), lr= self.learning_rate)
        self.alexnet_model= self.load_model()

    def load_model(self):
        num_classes = len (self.activities)
        num_ftrs= 4096
        self.alexnet_model.classifier[6]= nn.Linear(num_ftrs, num_classes)
        self.alexnet_model, optimizer= self.load_ckp('./models/alexnet.pth', self.alexnet_model, self.optimizer)
        self.alexnet_model.eval()
        return self.alexnet_model
    
    def load_ckp(self, checkpoint_fpath, model, optimizer):
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        valid_loss_min = checkpoint['valid_loss_min']
        return model, optimizer

    def predict_model(self, last):
        last= "./audio/" + str(last) + '.wav'
        spec= self.spec_to_image_3d(self.get_melspectrogram_db(last))
        device=torch.device('cpu')
        spec_t= torch.tensor(spec).to(device, dtype=torch.float32)
        spec_t= spec_t.permute(2,0,1)
        spec_t= spec_t.reshape(1,3,self.img_x,self.img_y)
        pr= self.alexnet_model.forward(spec_t)
        index = pr.argmax(dim=1).cpu().detach().numpy().ravel()[0]
        return index

    def spec_to_image_3d(self, spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        spec_scaled= cv2.cvtColor(spec_scaled,cv2.COLOR_GRAY2RGB)
        return spec_scaled

    def get_melspectrogram_db(self, file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
        wav,sr = librosa.load(file_path,sr=sr)
        if wav.shape[0]<5*sr:
            wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
        else:
            wav=wav[:5*sr]
        spec=librosa.feature.melspectrogram(y= wav, sr=sr, n_fft=n_fft, hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
        spec_db=librosa.power_to_db(spec,top_db=top_db)
        return spec_db        


        
