import numpy as np
import os
import math

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tfs
import skimage.io
import scipy.io

import torch.nn as nn
import torch.nn.functional as F

from channel_param import *

SCENARIO = "colo_cams"
IMG_DIR = f"../../data/single_scenario/{SCENARIO}"
BS_POS = (45, 0)

def estimate_pos(AoA_deg, path_delay, tx_pos):
    """
    return
    estimate coord, shape: (B, 2)
    """
    distance = (C * path_delay) / 2

    AoA_rad = torch.deg2rad(AoA_deg)

    x = tx_pos[0] + distance * torch.cos(AoA_rad)
    y = tx_pos[1] + distance * torch.sin(AoA_rad)

    x = torch.unsqueeze(x, 1)
    y = torch.unsqueeze(y, 1)
    out = torch.cat((x, y), axis=1)

    return out


def estimate_vec(doppler_shifts, aoa_degs):
    """
    Estimate user velocity in 2D using Doppler shifts and AoAs

    returns estimated velocity vector (vx, vy) in m/s
    """
    aoa_rads = np.radians(aoa_degs)

    # v * cos(theta) = fd * c / fc
    # this can be solved using LS
    A = np.vstack((np.cos(aoa_rads), np.sin(aoa_rads))).T
    b = np.array(doppler_shifts) * C / CR_FREQ

    # LS
    est_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return est_vec


def get_ground_truth_channel(UE_pos:torch.tensor, BS_pos=BS_POS) -> tuple:
    """
    Calculate channel states (AoA, path delay) based on ground turth UE position

    returns:
    AoA, path delay
    """
    # extract coord
    x_ue, y_ue = UE_pos[:, 0], UE_pos[:, 1]
    x_bs, y_bs = BS_pos
    dx = x_ue - x_bs
    dy = y_ue - y_bs

    # path delay
    dist = torch.sqrt(dx ** 2 + dy ** 2)
    path_delay = dist / C

    # Azimuth angle
    azimuth = torch.arctan2(dy, dx)

    return azimuth, path_delay


class UserTrackingDataset(Dataset):
    """
    dataset for loading RGB images and depth maps
    """
    def __init__(self, scenario="colo_cams", resize_dim=64,
                 wireless_mat_filename="colo_direct_wireless_dataset", 
                transforms=None, root_dir="../../data/single_scenario/"):
        
        # filename format: cam_{cam_id}_{x coord}_{y coord}.jpg
        self.filename_list = [] # key list : {x coord * 10000}_{y coord * 10}
        self.rgb_dm_filenames = {} # ({x coord * 10000}_{y coord * 10} : [rgb_filename, dm_filename])
        self.wireless_dict = {} # {x coord * 10000}_{y coord * 10} : [df index]

        # path stuff
        self.scenario = scenario
        self.root_dir = root_dir
        self.scenario_dir = os.path.join(self.root_dir, self.scenario)
        self.wireless_path = os.path.join(self.root_dir, wireless_mat_filename)

        # call
        self.read_images()
        self.wireless_df = process_mat(wireless_mat_filename)

        for i, data in self.wireless_df.iterrows():
            coord = data["loc"]

            _x = str(math.ceil(coord[0] * 10000.0))
            _xf = str(math.floor(coord[0] * 10000.0))
            _x0 = str(math.ceil(coord[0] * 100000.0))
            _xf0 = str(math.floor(coord[0] * 100000.0))

            if (_x == "0"):
                k_ceil = "00000_" + str(int(coord[1] * 10))
                k_floor = "00000_" + str(int(coord[1] * 10))

            elif (coord[0] < 1.0):
                comp = "0"
                if (coord[0] < 0.1):
                    comp += "0"
                
                k_ceil = comp + _x + "_" + str(int(coord[1] * 10))
                k_floor = comp + _xf + "_" + str(int(coord[1] * 10))
                k_ceil0 = comp + _x0 + "_" + str(int(coord[1] * 10))
                k_floor0 = comp + _xf0 + "_" + str(int(coord[1] * 10))

                self.wireless_dict[k_ceil0] = i
                self.wireless_dict[k_floor0] = i
                
            else:
                k_ceil = _x + "_" + str(int(coord[1] * 10))
                k_floor = _xf + "_" + str(int(coord[1] * 10))
                
            self.wireless_dict[k_ceil] = i
            self.wireless_dict[k_floor] = i


        # process image
        self.transforms = transforms
        if (transforms is not None):
            return
        
        self.resize_dim = resize_dim
        resize_trans = tfs.Resize((resize_dim, resize_dim))
        norm_trans = tfs.Normalize(mean=(0.306, 0.281, 0.251), std=(0.016, 0.0102, 0.013))
        self.transforms = tfs.Compose([
            tfs.ToPILImage(),
            resize_trans,
            tfs.ToTensor(),
            norm_trans,
        ])


    def read_images(self):
        rgb_path = os.path.join(self.scenario_dir, "rgb")
        rgb_filenames = os.listdir(rgb_path)

        # depth maps in .mat format
        depth_path = os.path.join(self.scenario_dir, "depth_maps")
        depth_map_filenames = os.listdir(depth_path)
       
        # process filename
        d1 = {}
        for fn in rgb_filenames:
            fns = fn[:-4].split("_")
            # make sure 4 digits
            if ("." in fns[2]):
                x_int, x_dig = fns[2].split(".")
                x_dig += "0" * (4 - len(x_dig))
                _x = "".join([x_int, x_dig])
            else:
                _x = fns[2] + "0000"
            
            if ('.' in fns[3]):
                y_int, y_dig = fns[3].split(".")
                _y = "".join([y_int, y_dig])
            else:
                _y = fns[3] + "0"
                
            k = _x + "_" + _y # remove digit
            d1[k] = fn

        d2 = {}
        for fn in depth_map_filenames:
            fns = fn[:-4].split("_")
            # make sure 4 digits
            if ("." in fns[2]):
                x_int, x_dig = fns[2].split(".")
                x_dig += "0" * (4 - len(x_dig))
                _x = "".join([x_int, x_dig])
            else:
                _x = fns[2] + "0000"
            
            if ('.' in fns[3]):
                y_int, y_dig = fns[3].split(".")
                _y = "".join([y_int, y_dig])
            else:
                _y = fns[3] + "0"
                
            k = _x + "_" + _y # remove digit
            d2[k] = fn
            
        for k in d1.keys():
            if (k not in d2):
                continue
            self.filename_list.append(k)
            self.rgb_dm_filenames[k] = [d1[k], d2[k]]
        

    def __len__(self):
        # must have rgb and depth_map files
        return len(self.filename_list)
    
    def __getitem__(self, index) -> dict:
        """
        returns
        - "img" : img_data,
        - "dm" : dm_data,
        - "channel" : wireless_data,
        - "coord" : coord,
        """
        k = self.filename_list[index]
        _files = self.rgb_dm_filenames[k]

        # process .jpg data
        jpg_filename = _files[0]
        jpg_filename = os.path.join(self.scenario_dir, "rgb", jpg_filename)
        img_data = skimage.io.imread(jpg_filename) # (720, 1280, 3)
        img_data = self.transforms(img_data) # (3, 224, 224)
    
        
        # process .mat file
        dm_filename = _files[1]
        dm_filename = os.path.join(self.scenario_dir, "depth_maps", dm_filename)
        
        mat_f = scipy.io.loadmat(dm_filename)
        dm_data = mat_f["depth"] # (720, 1280, 3)
        dm_data = self.transforms(dm_data) # (3, 224, 224)

        # process wireless data
        # AoA, single path delay, path gain, Doppler shift
        _aoa, _gain, _delay, _doppler = None, None, None, None

        if (k in self.wireless_dict):
            df_idx = self.wireless_dict[k]
            _loc = self.wireless_df.iloc[df_idx]["loc"]
            _chnl = self.wireless_df.iloc[df_idx]["channel"]

            # AoA estimation
            _chnl = _chnl.T
            spec, angles = music_algorithm(_chnl)
            spec_idx = np.argmax(spec)
            _aoa = angles[spec_idx]

            # path delay
            _delay = get_path_delay(_chnl)[0]

            # path gain
            _gain = get_channel_gain(_chnl) # (`N_ANN`)

            # Doppler shifts
            # past shifts
            Hs_past = [self.wireless_df.iloc[i]["channel"].T for i in range(max(df_idx-4, 0), max(df_idx, 4))]
            Hs_past = np.stack(Hs_past, axis=0)
            _doppler_past = get_doppler(Hs_past)

            # if df_idx < 5 pick 0 ~ 5
            # else pick df_idx-5 ~ df_idx
            Hs = [self.wireless_df.iloc[i]["channel"].T for i in range(max(df_idx-5, 0), max(df_idx, 5))]
            Hs = np.stack(Hs, axis=0)
            _doppler = get_doppler(Hs)

        else:
            print(f"[Warning]:  (key: {k}) (index: {index}) not found in wireless data")

        # user coord label (use from jpg)
        _, _, x, y = _files[0][:-4].split("_")
        coord = [float(x), float(y)]
        coord = torch.tensor(coord, dtype=torch.float32)

        _doppler_past_med = np.median(_doppler_past[:, 0])
        _doppler_med = np.median(_doppler[:, 0])
        
        _wireless_data = torch.tensor(
            np.array([_doppler_past_med, _doppler_med, _aoa, _delay[0]]), 
            dtype=torch.float32
        )

        _gain = torch.tensor(_gain, dtype=torch.float32)

        # dict triggers weird error
        return img_data, dm_data, \
                _wireless_data, _gain, \
                coord


class UserTrackingModel(nn.Module):
    """
    Using NN to refine user tracking with image data
    params:
    - `use_wireless`: use wireless channel feature 
    - `use_images`: use image and depth maps feature

    Output:
    predicted user position, channel states (AoA, delay, ...)
    """
    def __init__(self, n_wireless_features=6, img_size=64, output_size=4, 
                 use_wireless=True, use_images=True):
        
        # must use wireless or image features
        assert(use_wireless or use_images), "You should either enable use_wireless or use_images"

        super(UserTrackingModel, self).__init__()

        self.use_wireless = use_wireless
        self.use_images = use_images

        self.n_wireless_features = n_wireless_features
        self.img_size = img_size
        self.output_size = output_size

        # wireless data branch
        self.path_gain_fc = nn.Linear(N_ANN, 64)
        # AoA, path delay, doppler, past doppler
        self.wireless_fc = nn.Linear(64 + n_wireless_features, 32) # e.g.: 64 + 6 (aoa, path delay, doppler, past_doppler, est_pos)

        # CNN for image data
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.img_fc = nn.Linear(64 * 8 * 8, 64)

        # CNN for depth maps
        self.dm_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dmm_fc = nn.Linear(64 * 8 * 8, 64)

        # Combined branch
        combinsed_size = 160
        if (self.use_wireless and not self.use_images):
            combinsed_size = 32
        elif (self.use_images and not self.use_wireless):
            combinsed_size = 128

        self.combined_fc = nn.Linear(combinsed_size, 32)
        self.out_fc = nn.Linear(32, self.output_size)


    def forward(self, _wireless_data, gain, images, dm):
        """
        parameters:
        - `wireless_data`: [_doppler_past, _doppler, aoa, delay] (all of shape (B, ))
        - `gain`, : (B, N_ANN)
        - `images`, `dm` : (B, 3, 64, 64) RGB and depth maps
        """
        
        if (self.use_wireless):
            # process channel data
            _gain = self.path_gain_fc(gain)

            # intial estimate
            aoa, delay = _wireless_data[:, 2], _wireless_data[:, 3]
            # dp_past, dp_now = _wireless_data[:, 0], _wireless_data[:, 1]
            est_pos = estimate_pos(aoa, delay, BS_POS)
            # est_vec = estimate_vec([dp_past, dp_now])

            # _gain + doppler + aoa + delay
            _wl_input = torch.cat((_gain, _wireless_data, est_pos), dim=1)
            _wl_output = self.wireless_fc(_wl_input)

        if (self.use_images):
            # image
            _img_cnn = self.cnn(images)
            _img_cnn = _img_cnn.view(_img_cnn.size(0), -1) # flatten cnn output
            _img = self.img_fc(_img_cnn) # (B, 64)

            # depth maps
            _dm_cnn = self.dm_cnn(dm)
            _dm_cnn = _dm_cnn.view(_dm_cnn.size(0), -1)
            _dm = self.dmm_fc(_dm_cnn) # (B, 64)

        if (self.use_wireless and self.use_images):
            _comb_input = torch.cat((_wl_output, _img, _dm), dim=1)
        elif (self.use_wireless):
            _comb_input = _wl_output
        elif (self.use_images):
            _comb_input = torch.cat((_img, _dm), dim=1)

        _comb = self.combined_fc(_comb_input)
        output = self.out_fc(_comb)

        return output


def main():
    ds = UserTrackingDataset()
    print(len(ds))
    dataloader = DataLoader(ds, batch_size=16, shuffle=False)

    i = 0
    for batch in dataloader:
        i += 1
        img_data, dm_data, _wireless_data, _gain, coord = batch
        print(img_data.shape, dm_data.shape, _wireless_data.shape, _gain.shape, coord.shape) # [(B, 3, 64, 64), (B, 3, 64, 64), (B, 4), (B, 128), (B, 2)]

    
    # print("CUDA available:", torch.cuda.is_available()) # True

if __name__ == '__main__':
    main()