from predictor_model_alexnet import PredictorAlx
from predictor_model_densenet import PredictorDsn
from predictor_model_efficientnet import PredictorEfc
from predictor_model_resnet50 import PredictorRsn50
from predictor_model_resnet152 import PredictorRsn152
from predictor_model_VGG import PredictorVGG
from audio_handler import AudioHandler
import time
import os
from time import perf_counter
import logging

time_pred = 5
audio_database = 5
epochs= 2
predictions= []
model_no= 0

print('Enter the number for: 0-Alexnet, 1-Densenet, 2-Efficientnet, 3-VGG, 4-Resnet50, 5-Resnet152')
model_no = int(input())

if model_no == 0:
    model= PredictorAlx()
 
elif model_no == 1:
    model= PredictorDsn()
 
elif model_no == 2:
    model= PredictorEfc()
    
elif model_no == 3:
    model= PredictorVGG()
    
elif model_no == 4:
    model= PredictorRsn50()
    
elif model_no == 5:
    model= PredictorRsn152()
 
else:
    print("No model selected")
    quit()

audio= AudioHandler(time_pred, audio_database)
operative_count= 0

while (operative_count<= epochs):

    if operative_count== 0:
        className= model.__class__
        className= className.__name__
        logging.basicConfig(filename= className + '_app.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')               
        
    elab_time_start= perf_counter()

    last= audio.store_data_prediction()
    
    store_time= perf_counter() - elab_time_start
    
    logging.info('Start test iter no: %s', operative_count)    
    
    logging.info('Time elspased for extracting and store audio is %s seconds', store_time)
        
    #-----------------------------------------------
    
    if model_no== 0:
    
        time.sleep(1)
        
        elab_time_start= perf_counter()
        
        result= model.predict_model(last)
        
        elaboration_time= perf_counter() - elab_time_start
        
        if operative_count > 0:
            logging.info('Time elspased for elaborate a Alexnet prediction is %s seconds', elaboration_time)

    #--------------------------------------

    if model_no== 1:

        time.sleep(1)

        elab_time_start= perf_counter()
        
        result= model.predict_model(last)
        
        elaboration_time= perf_counter() - elab_time_start
        
        if operative_count > 0:   
            logging.info('Time elspased for elaborate a Densenet prediction is %s seconds', elaboration_time)

    
    #--------------------------------------
            
    if model_no== 2:           
    
        time.sleep(1)

        elab_time_start= perf_counter()
        
        result= model.predict_model(last)
        
        elaboration_time= perf_counter() - elab_time_start
        
        if operative_count > 0:    
            logging.info('Time elspased for elaborate a Efficientnet prediction is %s seconds', elaboration_time)

    
    #--------------------------------------
            
    if model_no== 3:      
    
        time.sleep(1)

        elab_time_start= perf_counter()
        
        result= model.predict_model(last)
        
        elaboration_time= perf_counter() - elab_time_start
        
        if operative_count > 0:    
            logging.info('Time elspased for elaborate a VGG prediction is %s seconds', elaboration_time)

    
    #--------------------------------------
            
    if model_no== 4:
    
        time.sleep(1)

        elab_time_start= perf_counter()
        
        result= model.predict_model(last)
        
        elaboration_time= perf_counter() - elab_time_start
        
        if operative_count > 0:    
            logging.info('Time elspased for elaborate a Resnet50 prediction is %s seconds', elaboration_time)

    #--------------------------------------
    
    if model_no== 5:      
    
        time.sleep(1)

        elab_time_start= perf_counter()
        
        result= model.predict_model(last)
        
        elaboration_time= perf_counter() - elab_time_start

        if operative_count > 0:
            logging.info('Time elspased for elaborate a Resnet152 prediction is %s seconds', elaboration_time)
    
    #--------------------------------------

    predictions.append(result)
    
    if (len (predictions)> audio_database):
        del prediction[0]

    operative_count+= 1
    
        

    
