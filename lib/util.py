from enum import Enum
import time
import math
import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw
from ctypes import *

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def softmax(x, axis=-1):
    max_val = x.max(axis=axis, keepdims=True)
    x = np.exp(x - max_val)
    return x / x.sum(axis=axis, keepdims=True)

def find_all_boxes(output, conf_thresh, num_classes, anchors):
    """
    Find bounding boxes from output tensor of Yolo-v2-tiny network.
    The size of the tensor for each 32*32 grid cell is num_anchors * (5 + num_classes).
    5 values are consisted of (centerx, centery, width, height, object_confidence)
    and num_classes values are confidence score for pre-selected classes.
    """
    anchors = np.array(anchors)

    b, _, h, w = output.shape
    num_anchors, _ = anchors.shape

    output = output.transpose((0, 2, 3, 1))
    output = output.reshape((b, h, w, num_anchors, num_classes+5))

    xs, ys, ws, hs, det_confs, class_predictions = np.split(
        output, indices_or_sections=[1, 2, 3, 4, 5], axis=-1)

    xs = sigmoid(xs)
    xs += np.arange(w).reshape((1, 1, w, 1, 1))
    xs /= w

    ys = sigmoid(ys)
    ys += np.arange(h).reshape((1, h, 1, 1, 1))
    ys /= h

    anchors = anchors.reshape((num_anchors, 2, 1))
    ws = np.exp(ws) * anchors[:, 0] / w
    hs = np.exp(hs) * anchors[:, 1] / h

    det_confs = sigmoid(det_confs)

    class_predictions = softmax(class_predictions)
    max_prediction_scores = class_predictions.max(axis=-1, keepdims=True)
    max_prediction_args = class_predictions.argmax(axis=-1, keepdims=True)

    results = np.concatenate(
        (xs, ys, ws, hs, det_confs, max_prediction_scores, max_prediction_args), axis=-1)
    results = results.reshape((b, -1, results.shape[-1]))

    all_boxes = []
    for boxes in results:
        boxes = boxes[boxes[:, 4]*boxes[:, 5] > conf_thresh]
        all_boxes.append(boxes)

    return all_boxes


def xywh2xxyy(xywh):
    w = xywh[2]/2
    h = xywh[3]/2
    return [xywh[0]-w, xywh[0]+w, xywh[1]-h, xywh[1]+h]


def get_box_area(xxyy):
    return (xxyy[1]-xxyy[0]) * (xxyy[3]-xxyy[2])


def calculate_iou(coord1, coord2):
    intersect_coord = [
        max(coord1[0], coord2[0]),
        min(coord1[1], coord2[1]),
        max(coord1[2], coord2[2]),
        min(coord1[3], coord2[3])
    ]
    if intersect_coord[0] >= intersect_coord[1] or \
            intersect_coord[2] >= intersect_coord[3]:
        return 0.

    intersect_area = get_box_area(intersect_coord)
    union_area = get_box_area(coord1) + get_box_area(coord2) - intersect_area

    return intersect_area / union_area


def nms(boxes, nms_thresh):
    """
    For all the candidate boxes, invalidate the box if the iou with more confident box exceeds nms_threshold.
    """
    num_boxes = boxes.shape[0]
    boxes = np.array(sorted(boxes, key=lambda box: box[4], reverse=True))
    box_filter = np.full(num_boxes, True)
    for i in range(num_boxes):
        for j in range(i+1, num_boxes):
            if calculate_iou(xywh2xxyy(boxes[i]), xywh2xxyy(boxes[j])) > nms_thresh:
                box_filter[j] = False
    return boxes[box_filter]


def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = [
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0]]

    def get_color(c, x, max_val):
        """ choose unique color for each class """
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = Draw(img)
    detections = []
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height
        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = int(box[6]+1e-5)
            detections += [(cls_conf, class_names[cls_id])]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.rectangle(
                [x1, y1 - 15, x1 + 6.5 * len(class_names[cls_id]), y1], fill=rgb)
            draw.text((x1 + 2, y1 - 13), class_names[cls_id], fill=(0, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)

    if savename:
        img.save(savename)
    return img


class RunMode(Enum):
    TORCH = 1
    C = 2
    CUDA = 3
    CUDAOptimized = 4


def run_mode_closure():
    """
    Create a closure functions to get or set global run_mode
    """
    run_mode = None

    def get_run_mode():
        return run_mode

    def set_run_mode(mode):
        nonlocal run_mode
        run_mode = mode

    return get_run_mode, set_run_mode


get_run_mode, set_run_mode = run_mode_closure()


def timer(func):
    from .mytorch.base import cu_dll
    """
    Decorator for measuring the time used
    """
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        if get_run_mode() == RunMode.CUDA or get_run_mode() == RunMode.CUDAOptimized:
            cu_dll.block_cpu()
        end_time = time.time()
        elapsed_time = (end_time - start_time)*1000
        print(f"{self} took {elapsed_time:.3f} ms to execute.")
        return result
    return wrapper


def layer_preproc(func):
    """
    Decorator for mytorch.functional
    In this project, intermediate data communications between functions is numpy.ndarray.
    This layer processes both input and output numpy.ndarray to appropriate form.

    TODO:
    Write about the functionality of layer_preproc at a report.
    In a report, you should include the following contents
        - A data structure that numpy.ndarray is transformed to
        - What data to allocate or free and why
        - Why this function is necessary for fair performance check
    """
    def wrapper(*args, **kwargs):
        proc_args = ()
        proc_kwargs = {}
        if get_run_mode() == RunMode.TORCH:
            import torch
            for arg in args:
                if type(arg) == np.ndarray:
                    proc_args += (torch.from_numpy(arg), )
                else:
                    proc_args += (arg, )
            for k in kwargs:
                if type(kwargs[k]) == np.ndarray:
                    proc_kwargs[k] = torch.from_numpy(kwargs[k])
                else:
                    proc_kwargs[k] = kwargs[k]
            ret = func(*proc_args, **proc_kwargs)
            return ret.numpy()
        
        elif get_run_mode() == RunMode.C:
            from .mytorch.base import c_dll

            for arg in args:
                if type(arg) == np.ndarray:
                    c_arg = {}
                    c_arg['pointer'] = arg.ctypes.data_as(POINTER(c_float))
                    c_arg['shape'] = arg.shape
                    proc_args += (c_arg, )
                else:
                    proc_args += (arg, )
            for k in kwargs:
                if type(kwargs[k]) == np.ndarray:
                    c_arg = {}
                    c_arg['pointer'] = kwargs[k].ctypes.data_as(POINTER(c_float))
                    c_arg['shape'] = kwargs[k].shape
                    proc_kwargs[k] = c_arg
                else:
                    proc_kwargs[k] = kwargs[k]
            
            c_output_p, output_shape = func(*proc_args, **proc_kwargs)
            np_out = np.ctypeslib.as_array(c_output_p, output_shape)
            out = np.copy(np_out)
            c_dll.c_free(c_output_p)
            return out
        
        elif get_run_mode() == RunMode.CUDA or get_run_mode() == RunMode.CUDAOptimized:
            from .mytorch.base import cu_dll

            cuda_free_list = []

            for arg in args:
                if type(arg) == np.ndarray:
                    cuda_arg = {}
                    cuda_arg['pointer'] = cu_dll.np2cuda(arg.ctypes.data_as(POINTER(c_float)), c_int(arg.size))
                    cuda_arg['shape'] = arg.shape
                    proc_args += (cuda_arg, )
                    cuda_free_list.append(cuda_arg['pointer'])
                else:
                    proc_args += (arg, )
            for k in kwargs:
                if type(kwargs[k]) == np.ndarray:
                    cuda_arg = {}
                    cuda_arg['pointer'] = cu_dll.np2cuda(kwargs[k].ctypes.data_as(POINTER(c_float)), c_int(kwargs[k].size))
                    cuda_arg['shape'] = kwargs[k].shape
                    proc_kwargs[k] = cuda_arg
                    cuda_free_list.append(cuda_arg['pointer'])
                else:
                    proc_kwargs[k] = kwargs[k]

            cuda_output_p, output_shape = func(*proc_args, **proc_kwargs)

            for cuda_pointer in cuda_free_list:
                cu_dll.cuda_free(cuda_pointer)
            c_output_p = cu_dll.cuda2np(cuda_output_p, c_int(np.prod(output_shape)))
            cu_dll.cuda_free(cuda_output_p)
            np_out = np.ctypeslib.as_array(c_output_p, output_shape)
            out = np.copy(np_out)
            cu_dll.c_free(c_output_p)
            return out
        
        return func(*args, **kwargs)
    return wrapper


def load_image(image_path):
    """
    Load image from path and resize for yolov2 network.

    Inputs:
    - image_path: image path to load from

    Returns:
    - image: resized image into (416,416)
    """
    image = Image.open(image_path).convert('RGB').resize((416,416))

    return image


def process_image(image):
    """
    Preprocessing image regarding the shape and normalization.

    Inputs:
    - image: the format of image should be RGB, and this function expects (H,W,C) dimensions.

    Returns:
    - np.array: preprocessed image with shape (1,C,H,W), normalized to range [0,1]
    """
    image = np.array(image, dtype=np.float32)
    image = image.transpose((2, 0, 1))
    image = image.reshape((1,)+image.shape)
    image /= 255.
    image = np.ascontiguousarray(image)
    return image

# memory layout for convolution is the following
#   bias, kernel
def load_conv_layer(loaded_weights, layer):
    oc, i, k, has_bn = layer
    kern_size = oc*i*k*k
    bias_size = 0 if has_bn else oc
    n_param = kern_size + bias_size
    
    weights = {}
    weights["bias"] = None if has_bn else loaded_weights[:bias_size]
    weights["kernel"] = loaded_weights[bias_size:n_param].reshape(oc, i, k, k)

    return loaded_weights[n_param:], weights

# memory layout for batchnorm is the following
#   beta, gamma, mean, var
def load_bn_layer(loaded_weights, layer):
    oc, _, _, _ = layer
    weights = {}
    weights["beta"] = loaded_weights[:oc]
    weights["gamma"] = loaded_weights[oc:oc*2]
    weights["mean"] = loaded_weights[oc*2:oc*3]
    weights["var"] = loaded_weights[oc*3:oc*4]

    return loaded_weights[4*oc:], weights

# load bn and conv layer
def load_layer(loaded_weights, layer):
    _, _, _, has_bn = layer
    weights = {}
    if has_bn:
        loaded_weights, weight = load_bn_layer(loaded_weights, layer)
        weights['bn'] = weight
    loaded_weights, weight = load_conv_layer(loaded_weights, layer)
    weights['conv'] = weight

    return loaded_weights, weights

def parse_weight(weight_path):
    loaded_weights = np.fromfile(weight_path, dtype=np.float32)
    # delete the first 4 that are not necessary
    loaded_weights = loaded_weights[4:]
    weights = []
    # oc, ic, k, has_bn
    layers = [
        [16, 3, 3, True],
        [32, 16, 3, True],
        [64, 32, 3, True],
        [128, 64, 3, True],
        [256, 128, 3, True],
        [512, 256, 3, True],
        [1024, 512, 3, True],
        [1024, 1024, 3, True],
        [125, 1024, 1, False]
    ]
    
    for layer in layers:
        loaded_weights, weight = load_layer(loaded_weights, layer)
        weights.append(weight)
    
    return weights