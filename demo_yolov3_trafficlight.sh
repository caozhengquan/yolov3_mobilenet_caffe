./build/yolo_det_trafficlight \
models/yolov3/mobilenet_yolov3_trafficlight_deploy.prototxt \
models/yolov3/mobilenet_yolov3_trafficlight_deploy_iter_8000.caffemodel \
 -file assets/ \
 -mean_value 0.5,0.5,0.5 -normalize_value 0.007843 -confidence_threshold 0.5 -cpu_mode gpu
