from lib.util import RunMode, set_run_mode, load_image, process_image, parse_weight, find_all_boxes, nms, plot_boxes
from lib.model import YOLOv2

config = {
    'run_mode': RunMode.TORCH,
    'image_path': './data/input/dog.jpg',
    'output_path': './prediction.jpg',
    'weight_path': './data/weight/yolov2-tiny-voc.weights',
    'conf_thresh': 0.5,
    'nms_thresh': 0.5,
    'class_names' : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'TVmonitor'],
    'anchors' : [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]]
}
a = 5
set_run_mode(config['run_mode'])

# load weight
yolo_weight = parse_weight(config['weight_path'])

# preprocessing
img = load_image(config['image_path'])
inp = process_image(img)

# intermediate network
output = YOLOv2(inp, yolo_weight)

# postprocessing
boxes = find_all_boxes(output, config['conf_thresh'], len(config['class_names']), config['anchors'])[0]
boxes = nms(boxes, config['nms_thresh'])
pred_img = plot_boxes(img, boxes, config['output_path'], config['class_names'])