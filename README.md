# AWN final project

Code for vision-aided joint channel parameter estimation and user tracking

- `channel_param.py` : initial channel parameter prediction
- `img_processing.py`: vision-aided channel refinement network
- `main_module.py` : main module integrating the training progress

## How to run
1. create a virtual environment
    - `conda create -n my_env python=3.10`
2. run `pip install -r requirements.txt`
3. for model training, run `main_module.py`
4. If you have trained a weight, put the weight into `weight_paths` list, and enable line 246 `plot_performance(...)`