import . from galaxia

#Library OS
import os

ROOT_DIR = os.path.abspath("app/lib")
sys.path.append(ROOT_DIR)  # To find local version of the library
from .mrcnn import utils
from .mrcnn import visualize
from .mrcnn.visualize import display_images
import .mrcnn.model as modellib
from .mrcnn.model import log
from .mrcnn.config import Config

import skimage.io
import matplotlib.pyplot as plt


config = galaxia.GalaxiaConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.75
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

class DetectionGalaxy():
    ROOT_DIR=""
    MODEL_DIR=""
    WEIGHTS_PATH=""
    config=None
    
    def __init__(self, ROOT_DIR="/work/work_teamULS/mask/Mask_RCNN-master", MODELOS_DIR="modelos",FileWeight="galaxia_all_1.h5"):
        self.ROOT_DIR= ROOT_DIR
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, MODELOS_DIR)
        self.WEIGHTS_PATH = os.path.join(self.MODEL_DIR,FileWeight)
        self.config = InferenceConfig()
        
        
    def get_ax(self, rows=1, cols=1, size=12):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
    
    def mean_pi (self, _img):
        myimg = cv2.imread(_img)
        avg_color_per_row = numpy.average(myimg, axis=0)
        avg_color = numpy.average(avg_color_per_row, axis=0)
        _r,_g,_b = avg_color[0],avg_color[1],avg_color[2]
        return(_r,_g,_b)
    
    def Detection(self, image, test=False):
        model = modellib.MaskRCNN(mode="inference",config=self.config,model_dir=self.MODEL_DIR)
        model.load_weights(self.WEIGHTS_PATH, by_name=True)
        class_names = ['BG',"S","E"]
        if(test):
            image=skimage.io.imread("../imagenes/GalaxyMany.jpg")
        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'], show_mask=False, ax=self.get_ax())
        return r