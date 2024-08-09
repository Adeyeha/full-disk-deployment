import pandas as pd
from deployment import FullDiskFlarePrediction
import os
#Issuing prediction for M1+ Flares
PATH1 = 'trained-models/Model_resnet_Epoch_27_fold1.pth'
fdp = FullDiskFlarePrediction(modelpath=PATH1,media_folder='resnet',modeltype='customresnet34',rgb=False)
data = pd.read_csv('pred_dates.csv')

# Assuming 'data' is your DataFrame and 'fdp' is your predictor object
output_file = 'predictions_resnet.txt'
headers = ["source_date","obs_date","raw_filename","noaa_ar_filename","local_request_date","error","flare_probability","non_flare_probability","explanation"]

# Check if the file exists and is not empty, if not, write the headers
if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
    with open(output_file, 'w') as file:
        file.write(','.join(headers) + '\n')

# Append prediction results
with open(output_file, 'a') as file:
    for idx, row in data.iterrows():
        print(f"Processing row {idx}")

        prediction = fdp.predict(
            date_=row['pred_date'],
            # path=r'E:\Comcast\Desktop\full-disk-deployment\media\raw\2014\01\06\2014_01_06__18_59_39_10__SDO_HMI_HMI_magnetogram.jp2',
            # save_artefacts=True,
            # generate_explain=True,
            # include_explain=False,
            save_artefacts=True,
            generate_explain=True,
            include_explain=False,
            explanation_layer=fdp.model.conv1
        )

        # Extract relevant prediction details
        source_date = prediction['source_date']
        obs_date = prediction['obs_date']
        raw_filename = prediction['raw_filename']
        noaa_ar_filename = prediction['noaa_ar_filename']
        local_request_date = prediction['local_request_date']
        error = prediction['error']
        flare_probability = float(prediction['flare_probability'])
        non_flare_probability = float(prediction['non_flare_probability'])
        explanation = prediction['explanation']

        # Write the prediction details to the file
        file.write(f"{source_date},{obs_date},{raw_filename},{noaa_ar_filename},{local_request_date},{error},{flare_probability},{non_flare_probability},{explanation}\n")

        # Optional: print the prediction details for debugging
        # print(source_date, obs_date, raw_filename, noaa_ar_filename, local_request_date, error, flare_probability, non_flare_probability, explanation)