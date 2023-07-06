import cv2
import rclpy
import torch
from std_msgs.msg import Header
from bbox_interfaces.msg import BoundingBox
from bbox_interfaces.msg import BoundingBoxes

def yolo2bboxs(output,names,colors,width,heigth,img_header:Header)->BoundingBoxes:
    """"Convert a yolov7 bounding box to a BoundingBoxes message for ROS2 
    params : 
        output : original yolov7 output
        names : dict of number : label
        colors : dict number color
        width and heigth : image dimension
        img_header : ros msg header

    returns : BoundingBoxes object

             """
    
    # Bounding box object
    res = BoundingBoxes()

    # header
    res.header = img_header

    for i in output:
        for *xyxy, conf, cls in reversed(i):
            bbox = BoundingBox()
            
            # confidence
            bbox.conf = float(conf)

            # top left corner
            bbox.x1 = int(xyxy[0])
            bbox.y1 = int(xyxy[1])

            # bottom rigth corner
            bbox.x2 = int(xyxy[2])
            bbox.y2 = int(xyxy[3])

            # label
            bbox.label = names[int(cls)]
            
            # color 
            bbox.color = colors[int(cls)]

            res.bboxes.append(bbox)

    res.img_width = width
    res.img_heigth = heigth

    return res

def yolov8tobboxs(output,names,colors,width,heigth)->BoundingBoxes:
    """"Convert a yolov7 bounding box to a BoundingBoxes message for ROS2 """

    res = BoundingBoxes()

    for results in output:
        for yolobbox in results.boxes:
            bbox = BoundingBox()

            # confidence
            bbox.conf = float(yolobbox.conf)
            
            # top left corner
            bbox.x1 = int(yolobbox.xyxy[0][0])
            bbox.y1 = int(yolobbox.xyxy[0][1])

            # bottom rigth corner
            bbox.x2 = int(yolobbox.xyxy[0][2])
            bbox.y2 = int(yolobbox.xyxy[0][3])

            # #label
            bbox.label = names[int(yolobbox.cls)]
            
            # color 
            bbox.color = colors[int(yolobbox.cls)]

            res.bboxes.append(bbox)
    res.img_width = width
    res.img_heigth = heigth

    return res

