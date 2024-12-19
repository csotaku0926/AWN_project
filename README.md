# AWN final project

Code for vision-aided joint channel parameter estimation and user tracking

- `channel_param.py` : initial channel parameter prediction
- `img_processing.py`: vision-aided channel refinement network
- `main_module.py` : main module integrating the training progress

## Preprocess
You should download the corresponding data used in "Colocated-Camera Scenario with Direct View" case in [ViWi dataset](https://www.viwi-dataset.net/scenarios.html) first

Put wireless channel data (a .mat file) in the path "{DATA_PATH}/{wireless_mat_filename}", where "DATA_PATH" defined in `channel_param.py` line 7, and "wireless_mat_filename" is the parameter of `UserTrackingDataset` `__init__`

Put image data in the path "{root_dir}/{scenario}/rgb", where "root_dir" and "scenario" are parameters of `UserTrackingDataset` `__init__`
Similarly, put depth map data in the path "{root_dir}/{scenario}/depth_maps"

## How to run
1. create a virtual environment
    - `conda create -n my_env python=3.10`
2. run `pip install -r requirements.txt`
3. for model training, run `main_module.py`
4. If you have trained a weight, put the weight into `weight_paths` list, and enable line 246 `plot_performance(...)`