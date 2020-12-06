# UNITER-ITM
This part of the repository is modify version of offcial implementation of [UNITER](https://github.com/ChenRocks/UNITER) and [VILLA](https://github.com/zhegan27/VILLA). And in this competition we focus on using pretrained single stream transformer and corresponeding ITM head to detect hateful memes.

![Overview of UNITER](https://convaisharables.blob.core.windows.net/uniter/uniter_overview_v2.png)

## Requirements

Before starting fine-tune the model, we first need to download the [pretrained weight](https://drive.google.com/file/d/1Amt8Iz6FpjHEkCDnraoMhOskID3qMxaZ/view?usp=sharing) to ``ROOT/pretrain_model``. The ZIP file will contained both uniter and villa pretrained weight taken from their office repo. The original villa pretrained weight can't use for fine-tune due to its missmatched static_dict namespace, hence I include updated version of villa \(ex: villa-large-fix.pt\) in the ZIP file.

After downloading you should see the following folder structure:

```bash
.
├── ERNIE-Vil
├── UNITER
├── VL-BERT
├── data
├── data_utils
├── pretrain_model
│   ├── ernie-vil
│   ├── faster_rcnn_r2_101_fpn_2x_img_clip
│   ├── uniter
│   │   ├── uniter-base.pt
│   │   ├── uniter-large.pt
│   │   ├── villa-base-fix.pt
│   │   └── villa-large-fix.pt
│   └── vl-bert
└── checkpoints
```

## Fine-tune on HatefulMeme

1. Convert annotation and RoI features we extract earlier to LMDB

    ```bash
    bash scripts/create_all_db.sh
    ```

2. Train the following uniter/villa models: \
    1 x uniter-large \
    1 x uniter-base \
    1 x villa-large \
    1 x villa-base \
    Each large model will take around 1 hour to train and base model will take around 20 min.

    ```bash
    bash scripts/train_uniter.sh
    ```

    After script is finshed. you shoulde see something like this in `ROOT/checkpoints`

    ```bash
    .
    ├── ERNIE-Vil
    ├── UNITER
    ├── VL-BERT
    ├── data
    ├── data_utils
    ├── pretrain_model
    └── checkpoints
        ├── ernie-vil
        ├── vl-bert
        └── uniter
            ├── uniter-base
            │   ├── log
            │   ├── final.pt
            │   ├── results
            │   ├── test.csv
            │   └── uniter-base.json
            ├── uniter-large-0
            │   ├── log
            │   ├── final.pt
            │   ├── results
            │   ├── test.csv
            │   └── uniter-large-0.json
            ├── villa-base-1
            │   ├── log
            │   ├── final.pt
            │   ├── results
            │   ├── test.csv
            │   └── villa-base-0.json
            └── villa-large-0
                ├── log
                ├── final.pt
                ├── results
                ├── test.csv
                └── villa-large-3.json
    ```
    `final.pt` are weights that is used to inference on the test set. \
    `test.csv` CSV files are the inference result.

## Rerun test set inference

```bash
bash scripts/rerun_test.sh
```

By runing this script it will use every checkpoint with `final.pt` filename in the path `ROOT/checkpoints/uniter/**/`. And the csv inference results will also be updated. This could be useful when you are directly working with the [fine-tuned checkpoints](https://drive.google.com/file/d/1j5_N_lMXs-LE93TofxZWsmjrOYzmZ8ov/view?usp=sharing).
