# yolo3_mobilenet

This is a pure C++ version YoloV3 and mobilenet detection implementation based on self-define caffe version. Now it is being opensourced. You are free to using this version to do some fantastic detection applications.

Inside this repo, we provide 2 usage of yolov3 mobilenet detection which are:

- 21 classes detection trained on VOC and COCO;
- trafficlight detection for both day and night, it accurate, light, and fast, almost can be CPU realtime(almost).


If you using our lib inside your project, pls consider star our repo. Also, our full code can be find on *[StrangeAI](http://strangeai.pro)*. 


## Build

to build this app, you gonna need some basic requirements:

- cuda 9.0 and cudnn 7.3;
- boost, gflags, protobuf(3.6.1), glog, [thor](http://github.com/jinfagang/thor);

that's all, if you have once build caffe, you shall already got all those requirements, just install *thor* lib additional.

## Run

For inference, just:

```
./demo_yolov3_voc.sh
```


## Result

Here are some result of our detection lib:

![](https://s1.ax1x.com/2018/12/27/F2vgT1.png)


![](https://s1.ax1x.com/2018/12/27/F2vGLj.jpg)
![](https://s1.ax1x.com/2018/12/27/F2vBSU.jpg)


## Copyright

the copyright of this project belongs to CTI Robotics. if you have any question, welcome to ask me via wechat: `jintianiloveu`.