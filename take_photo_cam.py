import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai as dai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs



# define pipeline
pipeline = dai.Pipeline()

# RGB camera
cam = pipeline.create(dai.node.ColorCamera)

cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

ispOut = pipeline.create(dai.node.XLinkOut)
ispOut.setStreamName('isp')


