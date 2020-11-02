import os
import skimage.io

class ImageServer(object):
    _width=0
    _height=0
    _ra=0
    _dec=0
    _scale=0
    _Config=""

    def __init__(self, Config):
        self._Config = Config
        verbose=0
        if (verbose==1):
            print("MODEL_DIR: ")
            print("LOG_DIR:")
        
    def setVariable(self, RA=0, DEC=0, SCALE=0, WIDTH=512, HEIGHT=512):
        self._ra = RA
        self._dec = DEC
        self._scale = SCALE
        self._width = WIDTH
        self._height = HEIGHT
    
    def getImageSkyServer(self,verbose=False):
        url = self._Config["url_skyserver"]+"ra="+str(self._ra)+"&dec="+str(self._dec)+"&scale="+str(self._scale)+"&width="+str(self._width)+"&height="+str(self._height)+""
        image=skimage.io.imread(url)
        if (verbose):
            print(url)
        return image
    
    def getImageIVOA(self,verbose):
        return None
    
    def getImage(self, Source="SKYSERVER", RA=0, DEC=0, SCALE=0.360115, WIDTH=512, HEIGHT=512, verbose=False):
        self.setVariable(RA, DEC, SCALE, WIDTH, HEIGHT)
        aux=Source
        if(aux=="SKYSERVER"):
            return self.getImageSkyServer(verbose)
        if(aux=="IVOA"):
            return self.getImageIVOA(verbose)

