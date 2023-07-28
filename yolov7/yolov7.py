#! /usr/bin/env python3

# Import ROS2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

# ROS2 messages
from sensor_msgs.msg import Image
from bbox_interfaces.msg import BoundingBox
from bbox_interfaces.msg import BoundingBoxes

import time
import os
import sys
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# To import the modules of the model
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

# Import for yolov7 model
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import letterbox
from utils.yolo2bbox import yolo2bboxs,yolov8tobboxs
from ultralytics import YOLO


class yolov8:
    def __init__(self,weigth):

        # Loading the model
        self.setup_model(weigth)


    def setup_model(self,weigth):
        """ Load the model and the parameters"""

        # Parameters
        # self.source = "./concours.jpg"
        self.weigth = weigth # "src/yolov7/weigths/empty_seats_yolov8.pt"
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = 'cpu'
        self.trace = 1
        self.augment = False
        self.warmup = True
        self.visualisation = True

        # Load model
        self.model = YOLO('src/yolov7/weigths/yolov8m.pt')
        self.model = YOLO(self.weigth)
    
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    
    def Inference(self,image:cv2):
        """ Run an inference on an image"""
        
        pred = self.model(image,save=False,conf=self.conf_thres,iou=self.iou_thres,show=False,verbose=False)

        return pred



# Class that contains the yolov7 model
# - Loads the model
# - Makes the inference
class yolov7:
    def __init__(self,weigth):
        # Loading the model
        self.setup_model(weigth)

    def setup_model(self,weigth):
        """ Load the model and the parameters"""

        # Parameters
        self.weigth = weigth
        # self.weigth = "src/yolov7/weigths/coco_merged_yolov7.pt"
        # self.weigth = "src/yolov7/weigths/empty_seats.pt"
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = [0]
        self.trace = 1
        self.augment = False
        self.warmup = True

        # Initialize
        set_logging()
        self.device = select_device()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weigth, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size

        # Adapt to the workspace
        if self.trace:
            self.model = TracedModel(self.model, self.device, self.img_size,"src/yolov7/yolov7")

        if self.half:
            self.model.half()  # to FP16
        
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def Inference(self,image:cv2):
        """ Run an inference on an image"""
        
        # Applying a mask to the image to filter the other objects
        # contour = np.array([[0,719],[0,450],[648,148],[1127,280],[971,719]]) 
        # mask    = np.zeros_like(image)
        # cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
        # image = cv2.bitwise_and(image, mask)

        # Padded resize
        img = letterbox(image, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        self.dataset = [(img,image)]

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()
        for img,im0s in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]) and self.warmup:
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.augment)[0]
                self.warmup = False

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0 = '',im0s
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    # if self.visualisation:
                    #     for *xyxy, conf, cls in reversed(det):
                    #         label = f'{self.names[int(cls)]} {conf:.2f}'
                    #         plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                
            return pred


# This class contains the ros interface wrapper for the detection model
# - Initializes the node
#- Creates the BBox publisher
# - Creates the Image suscriber
# - Receives the images, make the inference
# - Sends the resultsen groups=1, wei
class ros_interface(Node):

    def __init__(self,args):
        super().__init__("yolov7")

        # Getting arguments
        self.declare_parameter("topic",'bboxes')
        self.declare_parameter("weigth",'src/yolov7/weigths/coco_merged_yolov7.pt')
        self.declare_parameter("model",'yolov7')
        topic = self.get_parameter("topic").get_parameter_value().string_value
        self.model_use = self.get_parameter("model").get_parameter_value().string_value
        weigth = self.get_parameter("weigth").get_parameter_value().string_value

        # Creating the model
        if self.model_use=='yolov7':
            self.model = yolov7(weigth)
        elif self.model_use=='yolov8':
            self.model = yolov8(weigth)

        # Creating the bridge : ROS2  -> CV2 
        self.bridge = CvBridge()

        # Creating the publisher
        self.pub = self.create_publisher(BoundingBoxes,topic,10)

        # Creating the suscriber
        self.sub = self.create_subscription(Image, '/zed2/zed_node/rgb/image_rect_color', self.detection_callback,10)
        # self.sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.detection_callback,10)
        
    def detection_callback(self,img_msg:Image):
        """Callback function for the Image suscriber"""

        # Getting the image
        img = self.bridge.imgmsg_to_cv2(img_msg)

        # Making the inference
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        ouput = self.model.Inference(img)

        # Converting output to BBox
        msg = None
        if self.model_use=='yolov7':
            msg = yolo2bboxs(ouput,self.model.names,self.model.colors,img.shape[1],img.shape[0],img_header=img_msg.header)
        elif self.model_use=='yolov8':
            msg = yolov8tobboxs(ouput,self.model.names,self.model.colors,img.shape[1],img.shape[0],img_header=img_msg.header)

        # Publish results
        
        self.pub.publish(msg)


def ros_main(args=None): 
    """Launch the yolov7 node"""

    rclpy.init(args=args)
    node = ros_interface(args)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__=="__main__":
    ros_main()
