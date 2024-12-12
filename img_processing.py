import numpy as np
import h5py
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tfs
import skimage.io
import scipy.io

SCENARIO = "colo_cams"
IMG_DIR = f"../../data/single_scenario/{SCENARIO}"

class UserTrackingDataset(Dataset):
    """
    dataset for loading RGB images and depth maps
    """
    def __init__(self, scenario="colo_cams", transforms=None, root_dir="../../data/single_scenario/"):
        
        # filename format: cam_{cam_id}_{x coord}_{y coord}.jpg
        self.filename_list = [] # key list : {x coord}_{y coord}
        self.rgb_dm_filenames = {} # (cam_{cam_id}_{x coord}_{y coord} : [rgb_filename, dm_filename])

        # path stuff
        self.scenario = scenario
        self.root_dir = root_dir
        self.scenario_dir = os.path.join(self.root_dir, self.scenario)
        
        # call
        self.read_images()

        # process image
        self.transforms = transforms
        if (transforms is not None):
            return
        
        resize_trans = tfs.Resize((224, 224))
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
        
        # process filename
        for fn in rgb_filenames:
            fns = fn[:-4].split("_")
            k = fns[2] + "_" + fns[3]
            self.filename_list.append(k)
            self.rgb_dm_filenames[k] = [fn]

        # depth maps in .mat format
        depth_path = os.path.join(self.scenario_dir, "depth_maps")
        depth_map_filenames = os.listdir(depth_path)

        for fn in depth_map_filenames:
            fns = fn[:-4].split("_")
            k = fns[2] + "_" + fns[3]
            if (k in self.rgb_dm_filenames):
                self.rgb_dm_filenames[k].append(fn)


    def __len__(self):
        return len(self.rgb_dm_filenames)
    
    def __getitem__(self, index) -> dict:
        """
        returns
        - (img_data, dm_data)
        - img_data and dm_data are of (720, 1280, 3) shape
        """
        k = self.filename_list[index]
        _files = self.rgb_dm_filenames[k]

        # process .jpg data
        jpg_filename = _files[0]
        jpg_filename = os.path.join(self.scenario_dir, "rgb", jpg_filename)
        img_data = skimage.io.imread(jpg_filename) # (720, 1280, 3)
        img_data = self.transforms(img_data) # (3, 224, 224)
        
        if (len(_files) < 2):
            return img_data
        
        # process .mat file
        dm_filename = _files[1]
        dm_filename = os.path.join(self.scenario_dir, "depth_maps", dm_filename)
        
        mat_f = scipy.io.loadmat(dm_filename)
        dm_data = mat_f["depth"] # (720, 1280, 3)
        dm_data = self.transforms(dm_data) # (3, 224, 224)

        # user coord label
        x, y = k.split("_")
        coord = (float(x), float(y))
        coord = torch.tensor(coord)

        ret = {
            "img" : img_data,
            "dm" : dm_data,
            "coord" : coord,
        }

        return ret

def main():
    ds = UserTrackingDataset()
    img, dm, coord = ds[2]["img"], ds[2]["dm"], ds[2]["coord"]
    print(img.shape, dm.shape, coord.shape)

if __name__ == '__main__':
    main()