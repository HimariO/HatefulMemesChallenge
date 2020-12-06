from . import transforms as T
from albumentations import (
    BboxParams,
    Crop,
    Compose,
    ShiftScaleRotate,
    RandomBrightness,
    RandomContrast,
    RandomScale,
    Rotate,
    HorizontalFlip,
    MedianBlur,
)


def build_transforms(cfg, mode='train', norm_image=True):
    assert mode in ['train', 'test', 'val']
    min_size = cfg.SCALES[0]
    max_size = cfg.SCALES[1]
    assert min_size <= max_size

    if mode == 'train':
        flip_prob = cfg.TRAIN.FLIP_PROB
    elif mode == 'test':
        flip_prob = cfg.TEST.FLIP_PROB
    else:
        flip_prob = cfg.VAL.FLIP_PROB

    to_bgr255 = True

    normalize_transform = T.Normalize(
        mean=cfg.NETWORK.PIXEL_MEANS, std=cfg.NETWORK.PIXEL_STDS, to_bgr255=to_bgr255
    )

    # transform = T.Compose(
    #     [
    #         T.Resize(min_size, max_size),
    #         T.RandomHorizontalFlip(flip_prob),
    #         T.ToTensor(),
    #         normalize_transform,
    #         T.FixPadding(min_size, max_size, pad=0)
    #     ]
    # )
    bbox_params = BboxParams(
        format='pascal_voc',
        min_area=0, 
        min_visibility=0.2,
        label_fields=['fake_label'])
    album_augs = [
        HorizontalFlip(p=0.5),
        # RandomBrightness(limit=0.2, p=0.5),
        # RandomContrast(limit=0.2, p=0.5),
        RandomScale(scale_limit=(-0.3, 0.0), p=0.3),
        # MedianBlur(blur_limit=5, p=0.3),
        # Rotate(limit=30, p=0.25),
    ]
    album_augs = Compose(album_augs, bbox_params=bbox_params)

    if mode == 'train':
        all_augs = [
            T.Resize(min_size, max_size),
            T.ToTensor(),
            album_augs,
        ]
    else:
        all_augs = [
            T.Resize(min_size, max_size),
            T.ToTensor(),
        ]
    if norm_image:
        all_augs.append(normalize_transform)
    transform = T.Compose(all_augs)
    return transform
