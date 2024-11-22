# DATTNet<br />


Coder for "Enhancing Biventricular Segmentation with Multi-Scale UNet-Adapter and Consistency-Based Semi-Supervised Learning"<br />

## 0.Overview
![image](https://github.com/chz367/UNet-Adapter/blob/main/figure/overview.tif)

## 1.Environment<br />
Please prepare an environment with python 3.9 torch 1.12.1+cu113 and torchaudio 0.12.1+cu113.<br />

Please use the following code to install the required Python packages.
```python
    pip install -r requirements.txt
```
## 2. Datasets
The dataset in our study can be found in [ACDC](https://ieee-dataport.org/documents/automatic-cardiac-diagnosis-challenge).
After downloading this dataset, run the script dataset/cdc_dataprocessing. py to process the data and obtain the ACDC_RLV dataset required for this study.
```python
    acdc_data_processing.py 
```
## 3.Train/Test
Run the train script on ACDC_RLV dataset.

Train
```python
    train_model.py 
```

Test
```python
    python test_model.py
```
