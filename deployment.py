import pandas as pd
import numpy as np
import torch
from model.model import Custom_AlexNet
import torchvision.transforms as transforms 
from PIL import Image
from datetime import datetime, timezone
import requests
import re
import os
from noaa_ar_reader import NOAAExtractor
from post_hoc_analysis import get_attention_maps,superimpose_original
from io import BytesIO
# import cv2 as cv
import matplotlib.pyplot as plt


class FullDiskFlarePrediction:
    """
    Class to load and issue predictions for various flare prediction models.
    Assumes models require 512x512 8-bit full-disk magnetograms.
    Supports data sources: (i) Helioviewer API and (ii) French Mirror of Helioviewer API.
    Models are trained with PyTorch.
    """

    def __init__(self, modelpath):
        self.__modelpath = modelpath
        self.__setup_config()

    def __setup_config(self):
        """Set up configuration parameters."""
        self.__obs_date_pattern = re.compile(br'<DATE-OBS>(.*?)</DATE-OBS>')
        self.__source_date_pattern = re.compile(br'<DATE>(.*?)</DATE>')
        self.__filename_pattern = re.compile(r'filename="([^"]+)"')
        self.__media_folder = 'media'
        self.__request_uri = 'https://api.helioviewer.org/v2/getJP2Image/?date='
        self.__mirror_request_uri = 'https://helioviewer-api.ias.u-psud.fr//v2/getJP2Image/?date='
        self.__uri_encode = '&sourceId=19'
        self.meta = {
            'source_date': None, 
            'obs_date': None, 
            'raw_filename': None, 
            'noaa_ar_filename': None,
            'local_request_date': None,
            'error': None,
            'flare_probability': None,
            'non_flare_probability': None,
            'explanation': None
        }
        self.__input_hmi = None
        self.__model = None
        self.__include_explain = False
        self.__save_artefacts = False

    @staticmethod
    def __convert_date_format(date_str):
        """Convert date string to desired format."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").strftime('%Y-%m-%d %H:%M:%S')
    
    def __process_data(self, data):
        """Transform and process the input data for model prediction."""
        transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
        hmi = Image.open(BytesIO(data))
        hmi = transform(hmi).unsqueeze(0)
        return hmi

    def __extract_img_meta(self, local_request_date, response):
        """Extract metadata from the image."""
        self.meta['obs_date'] = self.__convert_date_format(self.__get_match(self.__obs_date_pattern, response.content))
        self.meta['source_date'] = self.__convert_date_format(self.__get_match(self.__source_date_pattern, response.content))
        self.meta['raw_filename'] = self.__get_match(self.__filename_pattern, response.headers['Content-Disposition'])
        self.meta['local_request_date'] = self.__convert_date_format(local_request_date)
        if not response.ok:
            self.meta['error'] = response.reason
        return True

    def __get_match(self, pattern, content):
        """Helper function to extract pattern matches."""
        match = pattern.search(content)
        if match:
            matched_content = match.group(1)
            # Check if content is bytes and decode if it is
            return matched_content.decode() if isinstance(matched_content, bytes) else matched_content.strip()
        return None

    def __save_hmi(self, response):
        """Save HMI data."""
        if response.ok:
            folder = os.path.join(self.__media_folder, "raw", *self.meta['raw_filename'].split('__')[0].split('_'))
            os.makedirs(folder, exist_ok=True)
            save_path = os.path.join(folder, self.meta['raw_filename'])
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False

    def __save_noaa_ar(self, df, filename):
        """Save NOAA AR data."""
        folder = os.path.join(self.__media_folder, "noaa_ar", *filename.split('_')[0:3])
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        df.to_csv(save_path, index=False)
        return True

    def __save_img_array(self, arr, image_type :str = None):
        """Save Image numpy array."""
        if arr is not None and image_type is not None:
            folder = os.path.join(self.__media_folder, image_type, *self.meta['local_request_date'].split(' ')[0].split('-'))
            os.makedirs(folder, exist_ok=True)
            save_path = os.path.join(folder, f"{self.meta['local_request_date'].replace('-','_').replace(' ','_').replace(':','_')}")
            np.save(save_path, arr)
            return True
        return False

    def __save_img(self, arr, extension='jpg', image_type:str=None):
        """Save JPG Image"""
        if arr is not None and image_type is not None:
            folder = os.path.join(self.__media_folder, image_type, *self.meta['local_request_date'].split(' ')[0].split('-'))
            os.makedirs(folder, exist_ok=True)
            save_path = os.path.join(folder, f"{self.meta['local_request_date'].replace('-','_').replace(' ','_').replace(':','_')}.{extension}")

            #plot overlayed image
            fig, ax = plt.subplots()
            im = ax.imshow(arr)
            plt.axis('off')
            fig.tight_layout(pad=0.1)
            fig.savefig(save_path, dpi=300, transparent=True)
            return True
        return False
        
    def __get_data(self):
        """Fetch and process the data for prediction."""

        # Initialize Active Region Extractor
        extractor = NOAAExtractor()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        final_date = now.replace(" ", "T") + "Z"
        
        # Fetch HMI Magnetogram
        for uri in [self.__request_uri, self.__mirror_request_uri]:
            response = requests.get(uri + final_date + self.__uri_encode)
            if response.ok:
                self.__input_hmi = self.__process_data(response.content)
                break
        
        # Get HMI magentogram metadata
        self.__extract_img_meta(final_date, response)
        if self.__save_artefacts == True:
            self.__save_noaa_ar(extractor.get_noaa_dataframe(), extractor.filename)
            self.meta['noaa_ar_filename'] = extractor.filename
            self.__save_hmi(response)
        return True

    def __predict(self):
        """Predict using the model."""
        try:
            self.__get_data()
            if self.__input_hmi is not None:
                device = torch.device('cpu')
                self.__model = Custom_AlexNet().to(device)
                checkpoint = torch.load(self.__modelpath, map_location=device)
                self.__model.load_state_dict(checkpoint['model_state_dict'])
                self.__model.eval()
                with torch.no_grad():
                    out = self.__model(self.__input_hmi)
                    noflare_prob,flare_prob = out[0].detach().numpy()
                    self.meta['flare_probability'], self.meta['non_flare_probability'] = flare_prob, noflare_prob
        except Exception as e:
            self.meta['error'] = str(e)
            raise
        return True

    def __explain(self):

        """Run explanation function"""
        guidedgradcam,original = get_attention_maps(self.__model,self.__input_hmi,self.meta['flare_probability'])

        if self.__save_artefacts == True:
            self.__save_img_array(guidedgradcam, "guidedgradcam")
            self.__save_img_array(original, "original")
            # self.__save_img(superimpose_original(original,guidedgradcam),image_type="superimposed")

        if self.__include_explain == True:
            self.meta['explanation'] = {
                'original':original,
                'guidedgradcam':guidedgradcam
            }

        return True

    def predict(self,include_explain=True,save_artefacts=False):
        """Predict using the model.""" 
        self.__include_explain = include_explain
        self.__save_artefacts = save_artefacts
        self.__predict()
        self.__explain()
        # self.input_hmi = np.transpose(self.__input_hmi.detach().numpy().squeeze(0), (1, 2, 0))
        return self.meta