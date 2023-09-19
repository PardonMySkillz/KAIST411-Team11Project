from .mytorch import functional as F

def YOLOv2(activation, weight):
    """
    TODO
    Implement yolo network using given mytorch function and weights.
    
    activation: An array containing input data, of shape (N, C, H, W)
    weight: An array containing per layer weights.
    The parsed weight structure is given below.
    weight = [{
        'conv': {
            'kernel'    : np.ndarray
            'bias'      : None if there is no bias else np.ndarray
        },
        'bn': {
            'mean'      : np.ndarray
            'var'       : np.ndarray
            'gamma'     : np.ndarray
            'beta'      : np.ndarray
        }
    }, ...]

    Note that at last layer, there is no batch normalization layer.
    Also, to implement the conv2d returning the same size as the input, you should pad before applying convolution network.

    For specific layer configuration, refer to `./references/yolov2-tiniy-voc.cfg` or `./references/yolo2-tiniy-voc.onnx`.
    """
    pass