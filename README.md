# TrackDiffuser: Nearly Model-Free Bayesian Filtering with Diffusion Model

## Overview
This repository provides the implementation of the **TrackDiffuser** framework, which utilizes diffusion models for Bayesian filtering. The provided scripts allow users to test the model.

## Directory Structure
```
|-- main_nonliearH_gaussian.py  # Run this script to obtain test results
|-- model/                      # Model-related files
|   |-- diffusion.py            # Diffusion model implementation
|   |-- temporal.py             # UNet network implementation
|-- results/                    # Results and datasets
|   |-- LOR/
|       |-- model3/
|           |-- nonlineargaussian/
|               |-- dataset_expF/
|                   |-- data_lor_v-20_r0.001_T20_100_nonlinear_gaussian_nonlinearF.pt  # Demo dataset
|               |-- r0.001/
|                   |-- checkpoint/
|                       |-- state_kitchen_partial_test2_best.pt  # Pretrained model
```

## Running the Code
To run the test, execute the following command:
```bash
python main_nonliearH_gaussian.py
```
This script will generate test results based on the provided dataset and pretrained model.

## Model Details
- **Diffusion Model**: Implemented in `model/diffusion.py`.
- **UNet Network**: Implemented in `model/temporal.py`.

## Dataset
The demo dataset is available in the following location:
```
results/LOR/model3/nonlineargaussian/dataset_expF/data_lor_v-20_r0.001_T20_100_nonlinear_gaussian_nonlinearF.pt
```

## Pretrained Model
The best-performing pretrained model is available at:
```
results/LOR/model3/nonlineargaussian/r0.001/checkpoint/state_kitchen_partial_test2_best.pt
```
This model can be used for inference and evaluation.

## Citation
If you use this code, please consider citing our work:
```
@inproceedings{XXX,
  author    = {XXX.},
  title     = {TrackDiffuser: Nearly Model-Free Bayesian Filtering with Diffusion Model},
  booktitle = {XXX},
  year      = {2025}
}
```

