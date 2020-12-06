import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob
from Mask_RCNN.mrcnn import utils
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.visualize import display_images
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn.model import log
from PIL import Image
import Mask_RCNN.facial_mask

if __name__=="__main__":

    print("Where is your face mask image?")
    img_path=input()

    custom_WEIGHTS_PATH="Facial_mask_best.h5"
    config=Mask_RCNN.facial_mask.Facial_maskConfig()
    class InferenceConfig(config.__class__):
        GPU_COUNT=1
        IMAGES_PER_GPU=1
    config=InferenceConfig()
    config.display()
    DEVIDE="/gpu:0"
    TEST_MODE="inference"
    with tf.device(DEVICE):
        model=modellib.MaskRCNN(mode="inference",model_dir="logs",config=config)

    model.load_weights(custom_WEIGHTS_PATH,by_name=True)
    pic=Image.open(img_path)
    if pic.getdata().mode=="RGBA":
        pic=pic.convert("RGB")
    image=np.array(pic.getdata()).reshape(pic.size[0],pic.size[1],3)
    np.transpose(image,(1,0,2))
    results=model.detect([image],verbose=1)
    r=results[0]
    mask_result=r['masks']
    size=(256,256,)
    mask_image=Image.new("RGB",size)
    pixels=mask_image.load()
    for i in range(size[0]):
        for j in range(size[1]):
            if(mask_result[i,j]==1):
                pixels[j,i]=(255,255,255)
    pic.save("../original.png")
    mask_image.save("../mask.png")
