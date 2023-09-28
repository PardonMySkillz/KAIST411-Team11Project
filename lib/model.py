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
    #in dictionary form
    kernel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']
    mean, var, gamma, beta = weight[0]['bn']['mean'], weight[0]['bn']['var'], weight[0]['bn']['gamma'], weight[0]['bn']['beta']
    print(activation.shape)
    print(type(activation))
    N, C, H, W = activation.shape
    print("len(weight)",len(weight))
    print("weight[0]", weight[0].keys())
    #1
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.leaky_relu(activation, 0.1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)

    #2
    activation = F.max_pool2d(activation, 2, 2)

    #3
    kernel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']
    mean, var, gamma, beta = weight[0]['bn']['mean'], weight[0]['bn']['var'], weight[0]['bn']['gamma'], weight[0]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)
    

    #4
    activation = F.max_pool2d(activation, 2, 2)

    #5
    kenrel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']
    mean, var, gamma, beta = weight[0]['bn']['mean'], weight[0]['bn']['var'], weight[0]['bn']['gamma'], weight[0]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)
    
    #6
    activation = F.max_pool2d(activation, 2, 2)

    #7
    kernel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']
    mean, var, gamma, beta = weight[0]['bn']['mean'], weight[0]['bn']['var'], weight[0]['bn']['gamma'], weight[0]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)

    #8
    activation = F.max_pool2d(activation, 2, 2)

    #9
    kernel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']
    mean, var, gamma, beta = weight[0]['bn']['mean'], weight[0]['bn']['var'], weight[0]['bn']['gamma'], weight[0]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)

    #10
    kernel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']

    #11
    kernel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']
    mean, var, gamma, beta = weight[0]['bn']['mean'], weight[0]['bn']['var'], weight[0]['bn']['gamma'], weight[0]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)

    #12
    activation = F.max_pool2d(activation,2, 1)

    #13
    kernel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']
    mean, var, gamma, beta = weight[0]['bn']['mean'], weight[0]['bn']['var'], weight[0]['bn']['gamma'], weight[0]['bn']['beta']
    activation = F.pad(activation, (1,1,1,1), 0)
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)

    #14
    kernel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']
    mean, var, gamma, beta = weight[0]['bn']['mean'], weight[0]['bn']['var'], weight[0]['bn']['gamma'], weight[0]['bn']['beta']
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.batch_norm(activation, mean, var, gamma, beta)
    activation = F.leaky_relu(activation, 0.1)

    #15
    kernel, bias = weight[0]['conv']['kernel'], weight[0]['conv']['bias']
    activation = F.conv2d(activation, kernel, bias, 1)
    activation = F.leaky_relu(activation, 0.1)

    return activation
    
   