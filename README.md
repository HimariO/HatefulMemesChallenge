# HatefulMemes

## Intro

This is the source code of FacebookAI HatefulMemes challenge solution...

## Dependency

* Docker >= 19.0.3
* [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

**NOTE:** Make sure you follow this [guide](https://docs.docker.com/engine/install/linux-postinstall/) to let docker run as root, so it can be run by shell scripts with out ```sudo```.

## System sepc

Original experiement was conduct on `GCP n1-highmem-16` instance init with `TensorFlow2.3/Keras.CUDA11.0.GPU` GCE Image:

* OS: Ubuntu 18.04.5 LTS
* CPU: 16 Core Intel CPU
* Memory: 104 GB
* GPU: 4 Nvidia T4
* Disk: 500GB HDD

Most of the data preprocessing and model training could be done with only 1 T4 GPU, except VL-BERT need 4 GPU to achieve high enough batch size when fine-tuning Faster-RCNN & BERT togather. \
**NOTE:** All models used in this project is using fp16 acceleration during training. Please use GPU support NVDIA AMP.

## Steps

1. Data preprocess and extract additional features. See detailed instruction at [data_utils/README](data_utils/README.md).

2. Train modified VL-BERT(2 large one and 1 base one). See detailed instruction at [VL-BERT/README](VL-BERT/README.md).

3. Train UNITER-ITM(1 large one and 1 base one) and VILLA-ITM(1 large one and 1 base one). See detailed instruction at [UNITER/README](UNITER/README.md).

4. Train ERNIE-Vil(1 large one and 1 base one). See detailed instruction at [ERNIE-VIL/README](ERNIE-Vil/README.md).

5. Ensemble by average predictions of all model then apply simple rule-base racism detector on top of it.

    ```bash
    bash run_ensemble.sh
    ```

    This script will let you select the predition of different model to taken into ensemble. As result it will output `ROOT/test_set_ensemble.csv` as final result and copy all the csv files used in ensemble to `ROOT/test_set_csvs` folder.