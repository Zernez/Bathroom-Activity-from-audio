# Bathroom audio activity classification feasibility

•	Sensor Fusion of Ambient and Wearable Sensors and Audio Classification for the Detection of Events of Daily Living technologies: Audio Classification using PyTorch and Librosa, MQTT/AQMP/HTTP, existing ambient gateways with sensors for sensor fusion and possible the TI Sensor Tag and/or an Android Wear device in this project.

•	Use an RPI unit to train an audio model to detect showering, toilet flushing and hand-wash activity. Report the accuracy of a generalized model and a model that is retrained at the setting (tested at least 10 different user bathrooms).

•	Investigate whether existing ambient and/or wearable sensors can be used to increase accuracy using sensor fusion. The system must be able to guide the user through learning activities for a localized learning experience, e.g., it should be possible to send an MQTT command to training in the current bathroom setting, then automatically retrain the model for this purpose and evaluate the improvement in accuracy.

•	 Also, it should investigate noise from the radio, etc. We will have access to at least 20 end-user homes for testing as part of two existing research projects. 

## Microphone equipment

Raspberry Pi4

ReSpeaker 2-Mics Pi HAT

## Lab-benchmark notebook

•	In "notebook" folder, a jupyter-notebook is placed. 

•	Audio .wav sample data are not present due to memeory-space requirements, a .csv file into "meta" folder contain the info for produce a dataset into the notebook. 

•	ETL provided in 1D (original Librosa output) transformed into 3D from mel-spectrum images.

•	Different models are implemented from older to newest known for image classification (Alexnet, VGG, Resnet50, Resnet152, DenseNet, EfficientNet). Models are pre-weighted from Pytorch ufficial settings, for reduce epochs.

•	In this notebook is possible to have a first glance of accuracy only from given dataset.

•	The mel-visualization in this notebook is "toilet_flush"

•	Provided function to record and save information for automatic ETL in a separate .csv file but marked in the same dataframe with folder #6

•	Accuracy provided over 90% with "Efficientnet"


## Labels selected

•	"vacuum_cleaner", "washing_machine", "brushing_teeth", "toilet_flush", "showering"

