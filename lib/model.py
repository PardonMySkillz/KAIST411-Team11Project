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
    #1
    kernel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']
    mean, var, gamma, beta = weight[0]['bn']['mean'], weight[0]['bn']['var'], weight[0]['bn']['gamma'], weight[0]['bn']['beta']
    N, C, H, W = activation.shape
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)
    activation = F.max_pool2d(activation, 2, 2) # (16, 208, 208)

    #2
    kernel, bias = weight[1]['conv']['kernel'], weight[1]['conv']['bias']
    mean, var, gamma, beta = weight[1]['bn']['mean'], weight[1]['bn']['var'], weight[1]['bn']['gamma'], weight[1]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)
    activation = F.max_pool2d(activation, 2, 2) # (32, 104, 104)
        
    #3
    kernel, bias = weight[2]['conv']['kernel'], weight[2]['conv']['bias']
    mean, var, gamma, beta = weight[2]['bn']['mean'], weight[2]['bn']['var'], weight[2]['bn']['gamma'], weight[2]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)
    activation = F.max_pool2d(activation, 2, 2) # (64, 52, 52)

    #4
    kernel, bias = weight[3]['conv']['kernel'], weight[3]['conv']['bias']
    mean, var, gamma, beta = weight[3]['bn']['mean'], weight[3]['bn']['var'], weight[3]['bn']['gamma'], weight[3]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)
    activation = F.max_pool2d(activation, 2, 2) # (128, 26, 26)

    #5
    kernel, bias = weight[4]['conv']['kernel'], weight[4]['conv']['bias']
    mean, var, gamma, beta = weight[4]['bn']['mean'], weight[4]['bn']['var'], weight[4]['bn']['gamma'], weight[4]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)
    activation = F.max_pool2d(activation, 2, 2) # (256, 13, 13)

    #6
    kernel, bias = weight[5]['conv']['kernel'], weight[5]['conv']['bias']
    mean, var, gamma, beta = weight[5]['bn']['mean'], weight[5]['bn']['var'], weight[5]['bn']['gamma'], weight[5]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)
    activation = F.max_pool2d(activation, 2, 1) # (512, 13, 13)

    #7
    kernel, bias = weight[6]['conv']['kernel'], weight[6]['conv']['bias']
    mean, var, gamma, beta = weight[6]['bn']['mean'], weight[6]['bn']['var'], weight[6]['bn']['gamma'], weight[6]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1) # (1024, 13, 13)

    #8
    kernel, bias = weight[7]['conv']['kernel'], weight[7]['conv']['bias']
    mean, var, gamma, beta = weight[7]['bn']['mean'], weight[7]['bn']['var'], weight[7]['bn']['gamma'], weight[7]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1) # (1024, 13, 13)

    #9
    # activation = F.pad(activation, (1,1,1,1), 0)
    kernel, bias = weight[8]['conv']['kernel'], weight[8]['conv']['bias']
    activation = F.conv2d(activation, kernel, bias, 1)

    return activation
    
   