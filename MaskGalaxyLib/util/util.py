import os
from datetime import datetime, timedelta
import time

def callWithNonNoneArgs(f, *args, **kwargs):
    kwargsNotNone = {k: v for k, v in kwargs.items() if v is not None}
    return f(*args, **kwargsNotNone)

def getName(ra,dec,server,Name):
    return Name.format(ra,dec,server,getNameDate())

def getNameDate():
    return datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]

def ClearFile(SecondsDel, PathImage, PathCatalog):
    try:
        now = datetime.now()
        for f in os.listdir(PathImage):
            filePath = os.path.getctime(os.path.join(PathImage,f))
            diff = (now - datetime.fromtimestamp(filePath)).total_seconds()
            Limit = timedelta(seconds = int(SecondsDel)).total_seconds()
            if (diff > Limit):
                os.remove(os.path.join(PathImage,f))
        for f in os.listdir(PathCatalog):
            filePath = os.path.getctime(os.path.join(PathCatalog,f))
            diff = (now - datetime.fromtimestamp(filePath)).total_seconds()
            Limit = timedelta(seconds = int(SecondsDel)).total_seconds()
            if (diff > Limit):
                os.remove(os.path.join(PathCatalog,f))
    except Exception as error:
        print(error)
        return None
    return None
    

