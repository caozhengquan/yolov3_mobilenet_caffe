# !/usr/bin/sh

./build/yolo_det_voc \
models/yolov3/mobilenet_yolov3_deploy.prototxt \
models/yolov3/mobilenet_yolov3_deploy_iter_58000.caffemodel \
 -file ./assets/v1.mp4 \
 -mean_value 0.5,0.5,0.5 -normalize_value 0.007843 -confidence_threshold 0.5 -cpu_mode gpu