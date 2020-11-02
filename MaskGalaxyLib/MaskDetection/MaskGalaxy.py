import sys
import os
#ROOT_DIR = os.path.abspath("Mask_RCNN-master")

ROOT_DIR = os.getcwd()
sys.path.insert(1, ROOT_DIR)

#sys.path.append("/home/ubuntu/webserver/Lib")
from .. import galaxia

#ROOT_DIR = "/work/work_teamULS/mask/Mask_RCNN-master"
#ROOT_DIR = "/home/ubuntu/webserver/Lib"
#sys.path.insert(1, ROOT_DIR)

dirPath = os.path.basename(ROOT_DIR)

sys.path.append(dirPath)  # To find local version of the library
#print(dirPath)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config

import skimage.io
import matplotlib.pyplot as plt

config = galaxia.GalaxiaConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5
    IMAGE_MIN_DIM = 2048
    IMAGE_MAX_DIM = 2048

class MaskGalaxy():
    ROOT_DIR=""
    MODEL_DIR=""
    WEIGHTS_PATH=""
    class_names = ['BG',"S","E"]
    RES=None
    IMAGE = None
    config = None


    def __init__(self, Config):
        self.config = InferenceConfig()
        
    def get_ax(self, rows=1, cols=1, size=12):
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
    
    def mean_pi (self, _img):
        myimg = cv2.imread(_img)
        avg_color_per_row = numpy.average(myimg, axis=0)
        avg_color = numpy.average(avg_color_per_row, axis=0)
        _r,_g,_b = avg_color[0],avg_color[1],avg_color[2]
        return(_r,_g,_b)
    
    def Detection(self,image, model1, test=False, ):
        self.image=image
        results = model1.detect([image], verbose=0)
        self.RES = results[0]
        return self.RES
    
    def getDetectionImage(self):
        return self.IMAGE
        
        