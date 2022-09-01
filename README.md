# Bathroom-Audio-Classification_Feasibility

•	Sensor Fusion of Ambient and Wearable Sensors and Audio Classification for the Detection of Events of Daily Living technologies: Audio Classification using PyTorch and Librosa, MQTT/AQMP/HTTP, existing ambient gateways with sensors for sensor fusion and possible the TI Sensor Tag and/or an Android Wear device in this project.

•	Use an RPI unit to train an audio model to detect showering, toilet flushing and hand-wash activity. Report the accuracy of a generalized model and a model that is retrained at the setting (tested at least 10 different user bathrooms).

•	Investigate whether existing ambient and/or wearable sensors can be used to increase accuracy using sensor fusion. The system must be able to guide the user through learning activities for a localized learning experience, e.g., it should be possible to send an MQTT command to training in the current bathroom setting, then automatically retrain the model for this purpose and evaluate the improvement in accuracy.

•	 Also, it should investigate noise from the radio, etc. We will have access to at least 20 end-user homes for testing as part of two existing research projects. 

## Microphone equipment

Lippa Trådløs Lavalier Mikrofon, USB-C

<img src="https://github.com/Zernez/Bathroom-Audio-Classification_Feasibility/blob/main/Microphone.jpg" width="200">

USB-A / USB-C KONVERTER / OTG ADAPTER XQ-ZH0011 - USB 3.0 - SORT

<img src="https://github.com/Zernez/Bathroom-Audio-Classification_Feasibility/blob/main/USB-A-C-Converter.jpg" width="180">

## Lab-benchmark notebook

•	In "notebook" folder, a jupyter-notebook is placed. 
•	Audio .wav sample data are not present due to memeory-space requirements, a .csv file into "meta" folder contain the info for produce a dataset into the notebook. 
•	ETL provided in 1D (original Librosa output) and 3D (RGB, needed for DenseNet and EfficientNet) from mel-spectrum images.
•	Different models are implemented from older to newest known for image classification (Resnet34, Resnet50, DenseNet201, EfficientNet_V2_L). Models are pre-weighted  from Pytorch ufficial settings, for reduce epochs.
•	In this notebook is possible to have a first glance of accuracy only from given dataset.

## Labels selected
•	vacuum_cleaner, water_drops, washing_machine, brushing_teeth, toilet_flush
•	Future candidate to discuss for inserting: Breathing, Showering, Footsteps, Open-Faucet

## TODO
•	Confusion-matrix needed, some feature maybe are not so pertinent e.g. "water-drops"
•	Feature for audio extract, labeling and appropriate merge into pre-existent database and dataframe.
•	Discovery of tuning some hyperparameters, e.g. learning rate and padding (maybe not useful in this case) 
•	Evaluation of possible usage of Google COLAB.
•	Extraction of audio data in different places and decision about metrics of evaluation.









