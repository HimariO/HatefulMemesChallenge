
## ERNIE-ViL

![ernie_vil_struct](.meta/ernie_vil_struct.png)  

This part of repo is the one-to-one reimplementation of the offical paddle implementation of [ERNIE-Vil](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-vil) I build for this competition. Since ERNIE-Vil is the best performaning vision-langage model on VCR leaderboard during the competition period, I didn't put to much effort to improve ERNIE-Vil's performance on hateful memes challenge. My main purpose of including this model is to add more diversity to the final ensemble result.

## Requirements

As usual, please download the [pretrained weight](https://drive.google.com/file/d/1GHuupFmwr2Voyw2h-c4TByxXTUJlMqW3/view?usp=sharing) to `ROOT/checkpoints` before going to next step. After you place the weight npz files to the right place, your project structure now shoule look like:

```bash
.
├── ERNIE-Vil
├── UNITER
├── VL-BERT
├── data
├── data_utils
├── pretrain_model
│   ├── uniter
│   ├── faster_rcnn_r2_101_fpn_2x_img_clip
│   ├── ernie-vil
│   │   ├── ernie-vil-large-vcr.npz
│   │   └── ernie-vil-large.npz
│   └── vl-bert
└── checkpoints
```

The pretrained weight is extracted from paddle checkpoint released in the original repo with `ROOT/ERNIE-Vil/extract_weight.py` script.

## Fine-tune on HatefulMeme

You can use the following script to train and inference on test set.

```bash
bash train_all.sh
```

After you run the `train_all.sh` shell script, you should see the following reuslts:

```bash
.
├── ERNIE-Vil
├── UNITER
├── VL-BERT
├── data
├── data_utils
├── pretrain_model
└── checkpoints
    ├── vl-bert
    ├── uniter
    └── ernie-vil
         ├── ernie-vil-large
         │   ├── args.pickle
         │   ├── ernie_vil.large.json
         │   ├── final.ckpt
         │   ├── lightning_logs
         │   ├── task_meme_gcp.json
         │   └── test_set.csv
         └── ernie-vil-large-vcr
            ├── args.pickle
            ├── ernie_vil.large.json
            ├── final.ckpt
            ├── lightning_logs
            ├── task_meme_gcp.json
            └── test_set.csv
```

`final.ckpt` are weights that is used to inference on the test set. \
`test_set.csv` CSV files are the inference result.\
`ernie_vil.*.json` JSON files are the model architecture config.\
`task_meme_gcp.json` JSON files are the task/dataset specific config.(in our case there is only one task unlike original implemntation)\
`args.pickle` pickle files are dump of all training parameters we get from argparser.

## Rerun test set inference

```bash
bash scripts/rerun_test.sh
```

By runing this script it will re-evaluate every checkpoint with `final.pt` filename in the path `ROOT/checkpoints/ernie-vil/**/`. And the csv inference results will also be updated. This could be useful when you are directly working with the [fine-tuned checkpoints](https://drive.google.com/file/d/1QSP0Vvwxb9OBHXGqqSYw16s5D10OIUWs/view?usp=sharing).
