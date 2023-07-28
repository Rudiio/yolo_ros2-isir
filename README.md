# YOLO ROS2

This is a ROS2 package that adapt Real-time objects detetor on ROS2. It is based on Yolos object detector.
The two supported models are Yolov7 and Yolov8. 
Yolov7 relies on this [paper](https://arxiv.org/abs/2207.02696) and the code for the model is originated from the official [implementation](https://github.com/WongKinYiu/yolov7).
The integration of yolov8 uses the official package of [ultralytics](https://github.com/ultralytics/ultralytics).

The pre-trained weigths were taken from the official repositories and you can have more informations about the training on the official repositories.

It was tested on Ubuntu 22 and ROS2 Humble.

## Installation

This package also uses a special interfaces that encodes 2 ROS2 messages:

```python
# detection confidence
float32 conf

# Object corners
uint16 x1
uint16 y1
uint16 x2
uint16 y2

# Object label
string label

# Object color
uint16[] color

# Position available
bool pos_available

# Object centroid position
float32[3] position
```

```python
# Headers
std_msgs/Header header
std_msgs/Header image_header

# Image Dimensions
uint16 img_width
uint16 img_heigth

# List of Bounding boxes
BoundingBox[] bboxes
```

``` shell
# cloning the repo into the ROS2 workspace
git clone https://github.com/Rudiio/yolo_ros2-isir.git

# Installing the requirements
pip install -r requirements.txt
pip install ultralytics

# building the package
colcon build
```

## Usage

When you use the package, you need to give it parameters : 
- the publishing topic
- the weigth path
- the model ('yolov7' or 'yolov8')

For example, in a launch file :
```python
launch_ros.actions.Node(
            package='yolov7',
            executable='yolov7',
            name='yolov8_empty_seats',
            parameters=[{'weigth' : 'src/yolov7/weigths/empty_seats_yolov8.pt'},
                        {'topic' : 'bboxes/empty_seats'},
                        {'model' : 'yolov8'}]),
```

The input topic (to suscribe) can be changed by remapping this topic : ```'/zed2/zed_node/rgb/image_rect_color'```
## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

```
@article{wang2022designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={arXiv preprint arXiv:2211.04800},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
