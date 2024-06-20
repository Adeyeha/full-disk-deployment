from deployment import FullDiskFlarePrediction
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv() 

# #Issuing prediction for M1+ Flares
# PATH1 = 'trained-models/alexnet-fold1.pth'
# fdp = FullDiskFlarePrediction(PATH1)
# pred = fdp.predict(
#     date_='2023-03-16 16:00:00',
#     # path=r'E:\Comcast\Desktop\full-disk-deployment\media\raw\2014\01\06\2014_01_06__18_59_39_10__SDO_HMI_HMI_magnetogram.jp2',
#     save_artefacts=False,
#     generate_explain=True,
#     include_explain=False
#     )
# # date_='2023-03-16 16:00:00', 

# print(pred)
# 
import os
from deployment import FullDiskFlarePrediction  # Replace with the actual module name

PATH1 = 'trained-models/alexnet-fold1.pth'
fdp = FullDiskFlarePrediction(PATH1,media_folder='alexnet')

# Base directory containing the subdirectories and files
base_dir = r'E:\Comcast\Desktop\full-disk-deployment\media\raw'

# Walk through the directory structure
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.jp2'):  # Check if the file is a JP2 file
            file_path = os.path.join(root, file)
            tic = time.time()
            # Predict using the model
            pred = fdp.predict(
                # date_='2023-03-16 16:00:00',
                path=file_path,
                save_artefacts=True,
                generate_explain=True,
                include_explain=False
            )
            tac = time.time()

            print(f"Processed file: {file} with {pred['flare_probability']} in {tac-tic}")

            # Print or handle the prediction result as needed
            # print(pred)
