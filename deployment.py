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

    def __init__(self, modeltype, modelpath, media_folder, rgb=False):
        self.__modelpath = modelpath
        self.__media_folder = media_folder
        self.__rgb = True
        self.__setup_config(modeltype)
        self.__model_config()

    # def __init__(self, modelpath,media_folder):
    #     self.__modelpath = modelpath
    #     self.__media_folder = media_folder
    #     self.__setup_config()

    def __checkmodel(self,modeltype):
        """
        Check if the model is valid.

        Raises:
        - ValueError: If the modeltype is not one of the valid models.
        """
        if modeltype not in NET_TYPES.keys():
            raise ValueError(f"parameter `modeltype` must be one of {' or '.join(NET_TYPES.keys())}")
        return True


    def __setup_config(self,modeltype):
        """Set up configuration parameters."""
        self.__obs_date_pattern = [re.compile(br'<DATE-OBS>(.*?)</DATE-OBS>'),re.compile(br'<DATE_OBS>(.*?)</DATE_OBS>'),re.compile(br'<DATE_OB>(.*?)</DATE_OB>')]
        self.__source_date_pattern = [re.compile(br'<DATE>(.*?)</DATE>')]
        self.__filename_pattern = [re.compile(r'filename="([^"]+)"')]
        self.__media_folder = 'media'
        self.__request_uri = 'https://api.helioviewer.org/v2/getJP2Image/?date='
        self.__mirror_request_uri = 'https://helioviewer-api.ias.u-psud.fr//v2/getJP2Image/?date='
        self.__uri_encode = '&sourceId=19'
        self.modeltype = modeltype if self.__checkmodel(modeltype) else None
        self.meta = {
            'source_date': None, 
            'obs_date': None, 
            'raw_filename': None, 
            'noaa_ar_filename': None,
            'local_request_date': None,
            'error': None,
            'flare_probability': None,
            'non_flare_probability': None,
            'explanation': None,
            'artefacts' : dict()
        }
        self.__input_hmi = None
        self.model = None
        self.__include_explain = False
        self.__save_artefacts = False
        self.__isfilepath = False

    # def __setup_config(self,):
    #     """Set up configuration parameters."""
    #     # self.__obs_date_pattern = re.compile(br'<DATE-OBS>(.*?)</DATE-OBS>')
    #     # self.__source_date_pattern = re.compile(br'<DATE>(.*?)</DATE>')
    #     # self.__filename_pattern = re.compile(r'filename="([^"]+)"')
    #     self.__obs_date_pattern = [re.compile(br'<DATE-OBS>(.*?)</DATE-OBS>'),re.compile(br'<DATE_OBS>(.*?)</DATE_OBS>'),re.compile(br'<DATE_OB>(.*?)</DATE_OB>')]
    #     self.__source_date_pattern = [re.compile(br'<DATE>(.*?)</DATE>')]
    #     self.__filename_pattern = [re.compile(r'filename="([^"]+)"')]
    #     self.__request_uri = 'https://api.helioviewer.org/v2/getJP2Image/?date='
    #     self.__mirror_request_uri = 'https://helioviewer-api.ias.u-psud.fr//v2/getJP2Image/?date='
    #     self.__uri_encode = '&sourceId=19'
    #     self.meta = {
    #         'source_date': None, 
    #         'obs_date': None, 
    #         'raw_filename': None, 
    #         'noaa_ar_filename': None,
    #         'local_request_date': None,
    #         'error': None,
    #         'flare_probability': None,
    #         'non_flare_probability': None,
    #         'explanation': None,
    #         'artefacts' : dict()

    #     }
    #     self.__input_hmi = None
    #     self.__model = None
    #     self.__include_explain = False
    #     self.__save_artefacts = False
    #     self.__isfilepath = False
    #     # self.__generate_explain = False

    def __model_config(self):
        """Configure the model"""
        if self.modeltype.lower() == "customalexnet":
            device = torch.device('cpu')
            self.model = Custom_AlexNet().to(device)

        elif self.modeltype.lower() == "customresnet34":
            device = torch.device('cpu')
            self.model = Custom_ResNet34().to(device)

        elif self.modeltype.lower() == "customvgg16":
            device = torch.device('cpu')
            self.model = Custom_VGG16().to(device)

        elif self.modeltype.lower() == "vgg16":
            device = torch.device('cpu')
            self.model = VGG16().to(device)

        checkpoint = torch.load(self.__modelpath, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    # @staticmethod
    # def __convert_date_format(date_str):
    #     """Convert date string to desired format."""
    #     try:
    #         return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d %H:%M:%S')
    #     except ValueError:
    #         return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def __convert_date_format(date_str):
        """
        Convert date string to desired format.
        Args:
            date_str (str): The date string to convert.
        Returns:
            str: The converted date string in the format '%Y-%m-%d %H:%M:%S'.
        """
        if date_str:
            formats = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]

            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    pass

            # Return the original string if no format matches
        return date_str

    @staticmethod
    def __extract_hmi_filename(path):
        """
        Extract the filename without the extension from a given path.

        Parameters:
        path (str): The file path from which to extract the filename.

        Returns:
        str: The filename without its extension.
        """
        # Get the basename from the path
        base = os.path.basename(path)
        
        # Split the basename into name and extension
        filename_without_extension = os.path.splitext(base)[0]
        
        return filename_without_extension

    def __process_data(self, data):
        """Transform and process the input data for model prediction."""
        transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
        if self.__isfilepath:
            hmi = Image.open(data)
        else:
            hmi = Image.open(BytesIO(data))
        hmi = transform(hmi).unsqueeze(0)
        return hmi

    def __extract_img_meta(self, local_request_date=datetime.now(timezone.utc), response=None):
        """Extract metadata from the image."""
        if not self.__isfilepath:
            self.meta['obs_date'] = self.__convert_date_format(self.__get_match(self.__obs_date_pattern, response.content))
            self.meta['source_date'] = self.__convert_date_format(self.__get_match(self.__source_date_pattern, response.content))
            self.meta['raw_filename'] = self.__get_match(self.__filename_pattern, response.headers['Content-Disposition'])
            self.meta['local_request_date'] = self.__convert_date_format(local_request_date)
            if not response.ok:
                self.meta['error'] = response.reason
        return True

    # def __get_match(self, pattern, content):
    #     """Helper function to extract pattern matches."""
    #     match = pattern.search(content)
    #     if match:
    #         matched_content = match.group(1)
    #         # Check if content is bytes and decode if it is
    #         return matched_content.decode() if isinstance(matched_content, bytes) else matched_content.strip()
    #     return None

    def __get_match(self, regex_patterns, content):
        """Helper function to extract pattern matches."""
        for pattern in regex_patterns:
            match = pattern.search(content)
            if match:
                matched_content = match.group(1)
                # Check if content is bytes and decode if it is
                return matched_content.decode() if isinstance(matched_content, bytes) else matched_content.strip()
        return None

    def __save_hmi(self, response):
        """Save HMI data."""
        if response is not None:
            if response.ok:
                folder = os.path.join(self.__media_folder, "raw", *self.meta['raw_filename'].split('__')[0].split('_'))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(folder, self.meta['raw_filename'])
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return save_path
                # return True
        return False

    def __save_noaa_ar(self, df, filename):
        """Save NOAA AR data."""
        folder = os.path.join(self.__media_folder, "noaa_ar", *filename.split('_')[0:3])
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        df.to_csv(save_path, index=False)
        return save_path
        # return True

    def __save_img_array(self, arr, image_type :str = None):
        """Save Image numpy array."""
        if arr is not None and image_type is not None:
            if not self.__isfilepath:
                folder = os.path.join(self.__media_folder, image_type, *self.meta['local_request_date'].split(' ')[0].split('-'))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(folder, f"{self.meta['local_request_date'].replace('-','_').replace(' ','_').replace(':','_')}")
            else:
                folder = os.path.join(self.__media_folder, image_type)
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(folder, self.__hmi_filename)
            np.save(save_path, arr)
            return f"{save_path}.npy"
            # return True
        return False

    def __save_img(self, arr, extension='jpg', image_type:str=None):
        """Save JPG Image"""
        if arr is not None and image_type is not None:
            if not self.__isfilepath:

                folder = os.path.join(self.__media_folder, image_type, *self.meta['local_request_date'].split(' ')[0].split('-'))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(folder, f"{self.meta['local_request_date'].replace('-','_').replace(' ','_').replace(':','_')}.{extension}")

            else:
                folder = os.path.join(self.__media_folder, image_type)
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(folder, self.__hmi_filename)

            #plot overlayed image
            fig, ax = plt.subplots()
            im = ax.imshow(arr)
            plt.axis('off')
            fig.tight_layout(pad=0.1)
            fig.savefig(save_path, dpi=300, transparent=True)
            return save_path
            # return True
        return False
        
    # def __get_data(self):
    #     """Fetch and process the data for prediction."""

    #     # Initialize Active Region Extractor
    #     extractor = NOAAExtractor()
    #     now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    #     final_date = now.replace(" ", "T") + "Z"
        
    #     # Fetch HMI Magnetogram
    #     for uri in [self.__request_uri, self.__mirror_request_uri]:
    #         response = requests.get(uri + final_date + self.__uri_encode)
    #         if response.ok:
    #             self.__input_hmi = self.__process_data(response.content)
    #             break
        
    #     # Get HMI magentogram metadata
    #     self.__extract_img_meta(final_date, response)
    #     if self.__save_artefacts == True:
    #         self.__save_noaa_ar(extractor.get_noaa_dataframe(), extractor.filename)
    #         self.meta['noaa_ar_filename'] = extractor.filename
    #         self.__save_hmi(response)
    #     return True

    def __get_data(self,date_=datetime.now(timezone.utc), path=None):
        """Fetch and process the data for prediction."""

        # Initialize Active Region Extractor
        extractor = NOAAExtractor()
        date_format =  "%Y-%m-%d %H:%M:%S" #os.getenv("date_format")

        if not self.__isfilepath:
            if isinstance(date_,str):
                date_ = datetime.strptime(date_,date_format)
            date_ = date_.strftime(date_format)
            final_date = date_.replace(" ", "T") + "Z"
            
            # Fetch HMI Magnetogram
            for uri in [self.__request_uri, self.__mirror_request_uri]:
                response = requests.get(uri + final_date + self.__uri_encode)
                if response.ok:
                    self.__input_hmi = self.__process_data(response.content)
                    break

            self.__extract_img_meta(final_date, response)

        else:
            self.__input_hmi = self.__process_data(path)
            response=None

        # Get HMI magentogram metadata
        if self.__save_artefacts == True:
            noaa_ar_save_path = self.__save_noaa_ar(extractor.get_noaa_dataframe(), extractor.filename)
            self.meta['noaa_ar_filename'] = extractor.filename
            magnetogram_save_path = self.__save_hmi(response)
            self.meta['artefacts'].update({
                "magnetogram":magnetogram_save_path,
                "noaa_ar":noaa_ar_save_path
                })
        return True

    # def __predict(self):
    #     """Predict using the model."""
    #     try:
    #         self.__get_data()
    #         if self.__input_hmi is not None:
    #             device = torch.device('cpu')
    #             self.__model = Custom_AlexNet().to(device)
    #             checkpoint = torch.load(self.__modelpath, map_location=device)
    #             self.__model.load_state_dict(checkpoint['model_state_dict'])
    #             self.__model.eval()
    #             with torch.no_grad():
    #                 out = self.__model(self.__input_hmi)
    #                 noflare_prob,flare_prob = out[0].detach().numpy()
    #                 self.meta['flare_probability'], self.meta['non_flare_probability'] = flare_prob, noflare_prob
    #     except Exception as e:
    #         self.meta['error'] = str(e)
    #         raise
    #     return True


    def __predict(self,date_=datetime.now(timezone.utc),path=None):
        """Predict using the model."""
        try:
            self.__get_data(date_,path)
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

    # def __explain(self):

    #     """Run explanation function"""
    #     guidedgradcam,original = get_attention_maps(self.__model,self.__input_hmi,self.meta['flare_probability'])

    #     if self.__save_artefacts == True:
    #         guidedgradcam_save_path = self.__save_img_array(guidedgradcam, "guidedgradcam")
    #         original_save_path = self.__save_img_array(original, "original")
    #         # self.__save_img(superimpose_original(original,guidedgradcam),image_type="superimposed")

    #     self.meta['artefacts'].update({
    #             'original':original_save_path,
    #             # 'deepshap':deepshap_save_path,
    #             # 'intgrad':intgrad_save_path,
    #             'guidedgradcam':guidedgradcam_save_path,
    #         })

    #     if self.__include_explain == True:
    #         self.meta['explanation'] = {
    #             'original':original,
    #             'guidedgradcam':guidedgradcam
    #         }

    #     return True

    def __explain(self,explanation_layer):

        """Run explanation function"""
        # guidedgradcam,deepshap,intgrad,original = get_attention_maps(self.model,self.__input_hmi,self.meta['flare_probability'], explanation_layer, self.modeltype, self.__rgb)
        guidedgradcam,deepshap,intgrad,original = get_attention_maps(self.__model,self.__input_hmi,self.meta['flare_probability'], explanation_layer, 'Alexnet', False)

        if self.__save_artefacts == True:
            guidedgradcam_save_path = self.__save_img_array(guidedgradcam, "guidedgradcam")
            deepshap_save_path = self.__save_img_array(deepshap, "deepshap")
            intgrad_save_path = self.__save_img_array(intgrad, "intgrad")
            original_save_path = self.__save_img_array(original, "original")
            # self.__save_img(superimpose_original(original,guidedgradcam),image_type="superimposed")
            self.meta['artefacts'].update({
                'original':original_save_path,
                'deepshap':deepshap_save_path,
                'intgrad':intgrad_save_path,
                'guidedgradcam':guidedgradcam_save_path,
            })

        if self.__include_explain == True:
            self.meta['explanation'] = {
                'original':original,
                'deepshap':deepshap,
                'intgrad':intgrad,
                'guidedgradcam':guidedgradcam,
            }

        return True

    # def predict(self,include_explain=True,save_artefacts=False):
    def predict(self, path=None, date_=datetime.now(timezone.utc), generate_explain=True, include_explain=True,save_artefacts=False,explanation_layer=None):
        """Predict using the model.""" 
        if path:
            self.__isfilepath = True
            self.__hmi_filename = self.__extract_hmi_filename(path)
        if include_explain:
            # self.__generate_explain = True
            self.__include_explain = include_explain
        self.__save_artefacts = save_artefacts
        self.__predict(date_,path)
        if generate_explain:
            self.__explain(None)
        # self.__explain()
        # self.input_hmi = np.transpose(self.__input_hmi.detach().numpy().squeeze(0), (1, 2, 0))
        return self.meta

    
