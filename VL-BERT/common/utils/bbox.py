import torch
import numpy as np
from PIL import (
    Image,
    ImageColor,
    ImageDraw,
    ImageFont,
    ImageOps)


def nonlinear_transform(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [k, 4] ([x1, y1, x2, y2])
    :param gt_rois: [k, 4] (corresponding gt_boxes [x1, y1, x2, y2] )
    :return: bbox_targets: [k, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-6)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-6)
    targets_dw = torch.log(gt_widths / (ex_widths).clamp(min=1e-6))
    targets_dh = torch.log(gt_heights / ((ex_heights).clamp(min=1e-6)))

    targets = torch.cat(
        (targets_dx.view(-1, 1), targets_dy.view(-1, 1), targets_dw.view(-1, 1), targets_dh.view(-1, 1)), dim=-1)
    return targets


def coordinate_embeddings(boxes, dim):
    """
    Coordinate embeddings of bounding boxes
    :param boxes: [K, 6] ([x1, y1, x2, y2, w_image, h_image])
    :param dim: sin/cos embedding dimension
    :return: [K, 4, 2 * dim]
    """

    num_boxes = boxes.shape[0]
    w = boxes[:, 4]
    h = boxes[:, 5]

    # transform to (x_c, y_c, w, h) format
    boxes_ = boxes.new_zeros((num_boxes, 4))
    boxes_[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    boxes_[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    boxes_[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes_[:, 3] = boxes[:, 3] - boxes[:, 1]
    boxes = boxes_

    # position
    pos = boxes.new_zeros((num_boxes, 4))
    pos[:, 0] = boxes[:, 0] / w * 100
    pos[:, 1] = boxes[:, 1] / h * 100
    pos[:, 2] = boxes[:, 2] / w * 100
    pos[:, 3] = boxes[:, 3] / h * 100

    # sin/cos embedding
    dim_mat = 1000 ** (torch.arange(dim, dtype=boxes.dtype, device=boxes.device) / dim)
    sin_embedding = (pos.view((num_boxes, 4, 1)) / dim_mat.view((1, 1, -1))).sin()
    cos_embedding = (pos.view((num_boxes, 4, 1)) / dim_mat.view((1, 1, -1))).cos()

    return torch.cat((sin_embedding, cos_embedding), dim=-1)


def bbox_iou_py_vectorized(boxes, query_boxes):
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    n_mesh, k_mesh = torch.meshgrid([torch.arange(n_), torch.arange(k_)])
    n_mesh = n_mesh.contiguous().view(-1)
    k_mesh = k_mesh.contiguous().view(-1)
    boxes = boxes[n_mesh]
    query_boxes = query_boxes[k_mesh]

    x11, y11, x12, y12 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x21, y21, x22, y22 = query_boxes[:, 0], query_boxes[:, 1], query_boxes[:, 2], query_boxes[:, 3]
    xA = torch.max(x11, x21)
    yA = torch.max(y11, y21)
    xB = torch.min(x12, x22)
    yB = torch.min(y12, y22)
    interArea = torch.clamp(xB - xA + 1, min=0) * torch.clamp(yB - yA + 1, min=0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou.view(n_, k_).to(boxes.device)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image, mode='RGBA')
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    # (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
    #                             ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                (left, top)],
            width=thickness,
            fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                    fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
                display_str,
                fill="black",
                font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.3):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    _colors = list(ImageColor.colormap.values())
    colors = []
    for c in _colors:
        if type(c) is str:
            colors.append(c + 'AA')
    # colors = [c + 'AA' for c in colors]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                16)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            xmin, ymin, xmax, ymax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                            int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image



