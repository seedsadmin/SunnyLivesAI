import argparse
import rasterio as rio
from rasterio.mask import mask

import geopandas as gpd
import geojson

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


from skimage import io
from scipy import ndimage as ndi
import cv2

from tqdm import tqdm
from pathlib import Path
import os

from azureml.core.model import Model

import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class Classifier(nn.Module):
    def __init__(self, name:str, num_classes:int):
        super(Classifier, self).__init__()

        if name == 'resnet101':
            self.encoder = models.resnet101(pretrained=True)

        if name == 'resnet50':
            self.encoder = models.resnet50(pretrained=True)

        if name == 'efficientnet-b3':
            self.encoder = EfficientNet.from_pretrained(name)

        if name == 'inceptionv3':
            self.encoder = models.inception_v3(pretrained=True)
            num_ftrs = self.encoder.AuxLogits.fc.in_features
            self.encoder.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)


        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes, bias=True)
        )

    def forward(self, images):
        return self.encoder(images)
    
def classify_roof(project_id, region_name, analytic_utm, projection):
    image_path = 'uploads/' + str(project_id)
    prod_output_str = 'uploads/' + str(project_id) +'/prodoutput'
    rsk_score = 'uploads/' + str(project_id) +'/rooftype'
    os.makedirs(rsk_score, exist_ok=True)
    
    INPUTS = [image_path, prod_output_str]
    OUTPUT = rsk_score
    REGION = region_name
    MODELNAME = 'inceptionv3'
    CHECKPOINT = 'models/rooftypeclassification'
    SIZE = 299
    UTM=analytic_utm
    src_utm=projection
    model = Classifier(name=MODELNAME, num_classes=7)
    model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    #load model checkpoint
    best_score = 0
    start_epoch = 0
    checkpoint_path = Path(Model.get_model_path(CHECKPOINT, version=2))
    if checkpoint_path.exists():
        print(f'Loading Checkpoint from {CHECKPOINT}')
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        state_dict = model.state_dict() 
        for key in state_dict:
            if key in loaded_dict:
                state_dict[key] = loaded_dict[key]

        model.load_state_dict(state_dict)
        start_epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']

        print(f'Loaded Checkpoint from: {CHECKPOINT}, \
                    with epoch {start_epoch} and best score {best_score}')
        
    RASTER = Path(INPUTS[0])
    VECTOR = Path(INPUTS[1])
    OUTPUT = Path(OUTPUT)
    print(INPUTS[0])

    footprints_file = f'{VECTOR}/{REGION}_simplified_footprints_from_model.geojson'
    print(f'Reading simplified footprints from {footprints_file}')
    footprints = gpd.read_file(f'{footprints_file}')
    #footprints.crs = UTM
    #footprints=footprints.to_crs(4326)

    print(footprints.crs)
    print(footprints.info)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #mapping the model output to rooft type id defined 
    #for ex in model RCC is predicted as 3 but RCC id defined is 1(in phase 1)
    dt_mapping = {
        0 : '2' ,
        1 : '3',
        2 : '4',
        3 : '1',
        4 : '7',
        5 : '5',
        6 : '6'
    }

    print('Predicting roof type')
    model.eval()
    building_dwell_type = []
    # for all the input files
    for raster_file in RASTER.glob('*.tif'):
        print(raster_file)
        with rio.open(raster_file) as rf:
            for idx, footprint in footprints.iterrows():
                try:
                    #crop the building from source image
                    building, building_transform = mask(rf, shapes=[footprint.geometry], 
                                                        crop=True)
                    _building = building.copy()
                    
                    #resize and normalise the data
                    _building = np.transpose(_building, (1, 2, 0))
                    _building = cv2.resize(_building, (SIZE, SIZE), 
                                            interpolation=cv2.INTER_NEAREST)
                    _building = np.asarray(_building, dtype='float32') / 255
                    for i in range(3):
                        _building[..., i] = (_building[..., i] - mean[i]) / std[i]
                    _building = torch.from_numpy(_building.transpose((2, 0, 1)).copy()).float()
                    _building = _building.unsqueeze(0)
                    
                    #run prediction
                    with torch.no_grad():
                        output = model(_building).cpu().detach().numpy()
                        label = np.argmax(output)
                        dwell_type = dt_mapping[label]

                        building_dwell_type.append({
                            'geometry': footprint.geometry,
                            'label': dwell_type
                        })
                        
                except Exception as e:
                    print(e)
                    continue
                    
    print(len(building_dwell_type))
    bdt_gdf = gpd.GeoDataFrame(building_dwell_type)
    print(bdt_gdf.head())
    bdt_gdf.groupby(['label']).agg(['count'])
    print(f'Saving the footprints')
    bdt_gdf.crs = src_utm
    bdt_gdf=bdt_gdf.to_crs(4326)
    bdt_gdf.to_file(f'{OUTPUT}/{REGION}_footprints.geojson', driver='GeoJSON')
    bdt_gdf.head()
