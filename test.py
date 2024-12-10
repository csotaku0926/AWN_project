import numpy as np
import h5py
import os
import pandas as pd

DATA_PATH = "../../data/single_scenario/data_generation_package/data"
N_IMG = 10

def process_mat(task_name:str) -> pd.DataFrame:
    """
    `task_name`: e.g., colo_direct_wireless_dataset

    read .mat file
    output pd.daraframe
    """
    wireless_dataset_path = os.path.join(DATA_PATH, f"{task_name}.mat")
    f = h5py.File(wireless_dataset_path, 'r')
    raw_key = list(f.keys())
    ds_obj = f[raw_key[1]][0][0]
    
    dataset = f[ds_obj]["user"]
    df = pd.DataFrame(columns=['loc', 'channel'])

    for i in range(N_IMG):
        user_i = f[dataset[i][0]]

        # location: (3, )
        loc_i = user_i["loc"][()]
        loc_i = np.array(loc_i).flatten()

        # channel: (# OFDM symbol, # antenna) of `numpy.void` type data (e.g.: by default (32, 128) size of (-6.51345623e-07, -8.99951125e-07))
        chnl_i = user_i["channel"][()]
        chnl_i = np.array(chnl_i)

        df.loc[i] = [loc_i, chnl_i]

    print(df.head())
    print(df.shape)

    return df


"""
TODO:
1. User tracking problem definition

                                image feature extraction V
channel parameters prediction --> sensing aided refinition (?) --> predicted channel states

- channel states: 
    - AoA and AoD -- AoA estimation, [MUSIC algorithm](https://github.com/MarcinWachowiak/music-aoa-estimation-py)
    - path gain -- compute norm of signal's multipath components
    - path delay -- FMCW radar principle (??)
    - Doppler shift -- relative velocity / signal strength
    - temporal channel correlation
     
- image data:
    - refine AoA or AoD, by aligning spatial features in the image
- output predicted user location, velocity
"""

if __name__ == '__main__':
    df = process_mat("colo_blocked_wireless_dataset")
    chnl = df.iloc[0]["channel"]
    print(chnl[0,0])
    print(type(chnl[0,0]))