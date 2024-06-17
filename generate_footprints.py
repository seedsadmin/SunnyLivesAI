import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as buildingcrop
from rasterio import features

import shapely
from shapely.geometry import shape, Polygon, Point
from shapely.affinity import affine_transform

from skimage import io
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

import numpy as np
import torch 
import torch.nn as nn
import torch.distributed as dist

import os 
from tqdm import tqdm
from pathlib import Path
import argparse
import cv2

from azureml.core.model import Model


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class UNet(nn.Module):
    def __init__(self, name='efficientnet-b0', pretrained=True, **kwargs):
        super(UNet, self).__init__()

        enc_sizes = {
            'efficientnet-b0': [16, 24, 40, 112, 1280],
            'efficientnet-b1': [16, 24, 40, 112, 1280],
            'efficientnet-b2': [16, 24, 48, 120, 1408],
            'efficientnet-b3': [24, 32, 48, 136, 1536],
            'efficientnet-b4': [24, 32, 56, 160, 1792],
            'efficientnet-b5': [24, 40, 64, 176, 2048],
            'efficientnet-b6': [32, 40, 72, 200, 2304],
            'efficientnet-b7': [32, 48, 80, 224, 2560],
            'efficientnet-b8': [32, 56, 88, 248, 2816]
        }

        encoder_filters = enc_sizes[name]
        decoder_filters = np.asarray([48, 64, 128, 160, 320]) 

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], 3, 1, stride=1, padding=0)

        self._initialize_weights()

        if pretrained:
            self.encoder = EfficientNet.from_pretrained(name)
        else:    
            self.encoder = EfficientNet.from_name(name)


    def extract_features(self, inp):
        out = []

        # Stem
        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(inp)))

        # Blocks
        for idx, block in enumerate(self.encoder._blocks):
            drop_connect_rate = self.encoder._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.encoder._blocks)
            y = block(x, drop_connect_rate=drop_connect_rate)
            if y.size()[-1] != x.size()[-1]:
                out.append(x)
            x = y
        
        # Head
        x = self.encoder._swish(self.encoder._bn1(self.encoder._conv_head(x)))
        out.append(x)

        return out


    def forward(self, x):
        batch_size, C, H, W = x.shape

        enc1, enc2, enc3, enc4, enc5 = self.extract_features(x)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))
        
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def transform_coords(polygon, transform):
    return affine_transform(polygon, [transform.a, transform.b, 
                                      transform.d, transform.e, 
                                      transform.xoff, transform.yoff])

def check_intersection(base, polygon):
    intersection = base.intersects(polygon)
    return intersection.any()
    
    

def gen_foot(project_id, region_name):
    INPUTS = ['uploads/' + str(project_id) + '/tiles']
    OUTPUT = 'uploads/' + str(project_id) +'/prodoutput'
    REGION = region_name
    MODELNAME = 'efficientnet-b6'
    CHECKPOINT = 'buildingsegmentation'
    IMAGESIZE = 512
    SCALES = [1, 3]
    DENSE = False

    print(torch.__version__)
    # os.makedirs(OUTPUT, exist_ok=True)
    INPUT = Path(INPUTS[0])
    OUTPUT = Path(OUTPUT)
    region_tiles = INPUT
    tile_paths = [str(p) for p in region_tiles.glob('*') if 'tif' in str(p)]
    num_tiles = len(tile_paths)
    print(f'Number of tiles to process {num_tiles}')
    CHECKPOINT = 'models/buildingsegmentation'

    if num_tiles:
        model = UNet(name=MODELNAME, pretrained=None)
        model = nn.DataParallel(model)
        print('model 1')
        if torch.cuda.is_available():
            model = model.cuda()
            print('model 2')
        checkpoint_path = Path(Model.get_model_path(CHECKPOINT, version=1))
        print('model 3')
        if checkpoint_path.exists():
            print('model 4')
            print("=> loading checkpoint '{}'".format(str(checkpoint_path)))
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            loaded_dict = checkpoint['state_dict']
            sd = model.state_dict()
            for k in model.state_dict():
                if k in loaded_dict:
                    sd[k] = loaded_dict[k]
            
            loaded_dict = sd
            model.load_state_dict(loaded_dict)
            best_score = checkpoint['best_score']
            start_epoch = checkpoint['epoch']
            print("loaded checkpoint from '{}' (epoch {}, best_score {})"
                    .format(CHECKPOINT, checkpoint['epoch'], checkpoint['best_score']))

    polygons = [] #footprints geometry
    values = [] #pixel values
    footprints_per_scale = {} #this is to store the lower scale footprints anf check if it intersects with higher scale
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    model.eval()
    print(f'Running prediction on {num_tiles}')

    count = 0
    for SCALE in SCALES:
        for index in tqdm(range(num_tiles)):
            try:
                tile_path = tile_paths[index]
                tile_name = tile_path.split('/')[-1]

                with rio.open(tile_path, 'r') as ref:
                    transform = ref.transform
                    crs = ref.crs

                    #read tile data
                    data = ref.read()
                    ref.close()
                    data = np.transpose(data, (1, 2, 0))

                    if DENSE:
                        data = data * 1.5

                    h, w = (np.asarray(data.shape[:2]) * SCALE).astype('int32')
                    scaled_image = cv2.resize(data, (w, h) , interpolation=cv2.INTER_NEAREST)
                    scaled_mask = np.zeros((h, w, 3)).copy()
                    for it in range(SCALE):
                        for jt in range(SCALE):
                            x = it * IMAGESIZE
                            y = jt * IMAGESIZE

                            image = scaled_image[y : y + IMAGESIZE, x : x + IMAGESIZE, :]
                            image = np.asarray(image, dtype='float32') / 255
                            for i in range(3):
                                image[..., i] = (image[..., i] - mean[i]) / std[i]

                            image = torch.from_numpy(image.transpose((2, 0, 1)).copy()).float()
                            image = image.unsqueeze(0)

                            with torch.no_grad():
                                pred = model(image.cuda(non_blocking=True))
                                out = torch.sigmoid(pred).cpu().numpy()
                                out = out[0, ...]
                                out = np.transpose(out, (1, 2, 0))
                                scaled_mask[y : y + IMAGESIZE, x : x + IMAGESIZE, ...] = out

                    mask = cv2.resize(scaled_mask, (IMAGESIZE, IMAGESIZE), interpolation=cv2.INTER_NEAREST)
                    mask = mask[..., 2] * (1 -  mask[..., 1]) * (1 - mask[..., 0]) #subtract boundary and contact point

                    #watershed method to seperate buildings
                    mask0 = (mask > 0.6)
                    local_maxi = peak_local_max(mask, indices=False, footprint=np.ones((11, 11)), 
                                                    labels=(mask > 0.4))
                    local_maxi[mask0] = True
                    seed_msk = ndi.label(local_maxi)[0]
                    y_pred = watershed(-mask, seed_msk, mask=(mask > 0.4), watershed_line=True)
                    y_pred = measure.label(y_pred, connectivity=2, background=0).astype('uint8')

                    #generate polygons from the predictions
                    polygon_generator = features.shapes(y_pred, mask=y_pred > 0)
                    for polygon, value in polygon_generator:
                        poly = shape(polygon).buffer(0.0)
                        if poly.area > 5:
                            ############################################################################
                            #This entire logic is written because model does not do great job at scale 1 
                            #when image resolution is not good and there is high building density 
                            #so at scale the generate result will be considered when scale 1 fails to detect the buildinf
                            ############################################################################
                            #for first scale directly add the polygons
                            if SCALE != SCALES[-1]:
                                polygons.append(transform_coords(poly, transform))
                                values.append(value) 
                            else:
                                # check if building not detected already
                                base = footprints_per_scale[SCALES[0]]['geometry'] #this is footprints generated at first scale
                                if not check_intersection(base, poly):
                                    polygons.append(transform_coords(poly, transform))
                                    values.append(value)


                footprints_per_scale[SCALE] = gpd.GeoDataFrame({'geometry': polygons})
            except Exception as e:
                print(e, index)

    num_buildings = len(polygons)
    print(f'Number of buildings detected {num_buildings}')
    
    os.makedirs(str(OUTPUT), exist_ok=True)
    if num_buildings:
        output_file = f'{str(OUTPUT)}/{REGION}_footprints_from_model.geojson'
        print(f'Writing output to {output_file}')
        polygon_gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values}, 
                                            crs=crs.to_wkt())
        polygon_gdf.to_file(f'{output_file}', driver='GeoJSON')
    
    print(polygon_gdf.head())