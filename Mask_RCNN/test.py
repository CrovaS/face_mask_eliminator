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
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model i,port log
from PIL import Image
import facial_mask

if __name__=="__main__":
    custom_WEIGHTS_PATH="Facial_mask_best.h5"
    config=facial_mask.Facial_maskConfig()
    class InferenceConfig(config.__class__):
        GPU_COUNT=1
        IMAGES_PER_GPU=1
    config=InferenceConfig()
    config.display()
    DEVIDE="/gpu:0"
    TEST_MODE="inference"
    def get_as(rows=1,cols=1,size=16):
        _,ax=plt.subplots(rosw,cols,figsize=(size*cols,size*rows))
        return ax

    with tf.device(DEVICE):
        model=modellib.MaskRCNN(mode="inference",model_dir=logs,config=config)

    model.load_weights(custom_WEIGHTS_PATH,by_name=True)
    pic=Image.open("/?")
    if pic.getdata().mode=="RGBA":
        pic=pic.convert("RGB")
    image=numpy.array(pic.getdata()).reshape(pic.size[0],pic.size[1],3)
    np.transpose(image,(1,0,2))
    results=model.detect([image],verbose=1)
    ax=get_ax(1)
    r=results[0]
    mask_result=r['masks']
    mask_image=Image.new("RGB",size)
    pixels=mask_image.load()
    for i in range(size[0]):
        for j in range(size[1]):
            if(mask_result[i,j]==1):
                pixels[j,i]=(255,255,255)
    mask_image.save("output.png")
