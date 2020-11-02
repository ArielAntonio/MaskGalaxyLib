import sys
import os
import skimage.io
from pathlib import Path
import matplotlib.pyplot as plt


# Library Mask R-CNN
ROOT_DIR = os.getcwd()
sys.path.insert(1, ROOT_DIR)
from . import galaxia
dirPath = os.path.basename(ROOT_DIR)
sys.path.append(dirPath)
from .mrcnn import utils
from .mrcnn import visualize
from .mrcnn.visualize import display_images
from .mrcnn.model import log
from .mrcnn.config import Config
import mrcnn.model as modellib
#*******************
from ImageServer.ImageServer import ImageServer
from CFound.CentroidImage import CentroidImage
from ConvertRaDec.ConvertCoord import ConvertCoord


config = galaxia.GalaxiaConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.75
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

# search galaxy on image, through red Mask r-cnn
class ClassMaskGalaxy:
    MODEL_DIR="" #path model dir
    WEIGHTS_PATH="" #path weight file 
    model=None
    RES=None
    image=None

    def __init__(self, _WEIGHTS_PATH, verbose=False):
        MODEL_DIR = os.path.dirname(Path(__file__).parent.absolute())
        WEIGHTS_PATH = _WEIGHTS_PATH

        config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference",config=config,model_dir=MODEL_DIR)
        self.model.load_weights(WEIGHTS_PATH, by_name=True)
        self.model.keras_model._make_predict_function()

        if verbose:
            print("MODEL_DIR: "+MODEL_DIR)
            print("WEIGHTS_PATH: "+WEIGHTS_PATH)

    def DetectGalaxy(self, image):
        self.image=image
        results = self.model.detect([image], verbose=0)
        self.RES = results[0]
        return self.RES

    def ShowImageDetection(self,image,r):
        class_names = ['BG',"S","E"]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'], show_mask=True, ax=self.__get_ax())

    def getCatalog(self,image,r, _RA, _DEC, _scale=0.360115, _WIDTH=512):
        url_votable="http://skyserver.sdss.org/dr15/SkyServerWS/ConeSearch/ConeSearchService?"
        CF = CentroidImage(image)
        df = CF.find_ListCentroid(r)
        CC = ConvertCoord(_WIDTH,df,_RA,_DEC,_scale,{'url_skyserver_votable':url_votable})
        return CC.getConvertCoord(ReturnOpt="df")

    def getCentroid(self,image,r):
        CF = CentroidImage(image)
        df = CF.find_ListCentroid(r)
        CF.CreateCatalog(df,r,"test.dat")
        return CF

    def GetImage(self,_RA=0,_DEC=0,_SCALE=0.360115, _WIDTH=512, _HEIGHT=512,server="Skyserver", _verbose=False):
        url_prefix="http://skyserver.sdss.org/dr15/SkyServerWS/ImgCutout/getjpeg?"
        IS = ImageServer({"url_skyserver":url_prefix})
        server = server.upper()
        return IS.getImage(Source=server ,RA=_RA, DEC=_DEC, SCALE=_SCALE, WIDTH=_WIDTH, HEIGHT=_HEIGHT, verbose=_verbose)

    def __mean_pi (self, _img):
        myimg = cv2.imread(_img)
        avg_color_per_row = numpy.average(myimg, axis=0)
        avg_color = numpy.average(avg_color_per_row, axis=0)
        _r,_g,_b = avg_color[0],avg_color[1],avg_color[2]
        return(_r,_g,_b)

    def __get_ax(self, rows=1, cols=1, size=12):
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
