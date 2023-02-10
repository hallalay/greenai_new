import rasterio
from rasterio.plot import show
from rasterio.windows import Window
import rasterio.mask
import numpy
import numpy as np
import cv2
from sentinelhub import BBox,bbox_to_dimensions


from keras.models import load_model

from main import getPrediction

from shapely import geometry



fp = r'green_ai\komprimerad_Orto2019EPSG3009.tif'

class ortophoto():
    '''
    Loads ortophoto
    
    @param path: path to ortophoto (GeoTif)
    '''
    def __init__(self, orto_path, model_path1, model_path2) -> None:
        self.orto = rasterio.open(orto_path) #Loads image
        self.model1 = load_model(model_path1,compile=False)
        self.model2 = load_model(model_path2,compile=False)

    def crop_poly(self, coords: list) -> numpy.ndarray:
        '''
        Crops the image into a polygon with given corners.

        @param coords: coords of the polygons corners.
        '''

        pix_coords = []
        for coord in coords:
            py, px = self.orto.index(coord[1],coord[0])
            # py = py-self.u_l[1] 

            pix_coords.append((px,py))
        
        pts = np.array(pix_coords)


        # print('pts: ', pts)

        ## crop the bounding rect
        rect = cv2.boundingRect(pts)

        x,y,w,h = rect
        poly = geometry.Polygon([[p[1], p[0]] for p in coords])


        window = Window(x,y,w,h)
        im = self.orto.read(window=window).T

        out_image,out_transform=rasterio.mask.mask(self.orto,[poly], crop=True)

        out_meta = self.orto.meta


        out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

        with rasterio.open("RGB.byte.masked.png", "w", **out_meta) as dest:
            
            im = out_image.T

        cv2.imwrite('square.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        self.poly_area = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


    def pred(self):
        self.gyf = getPrediction(self.poly_area, self.model1, self.model2)
        


    


# lat, lng = 59.27392143148289, 15.206654796775117

# coord = [lng,lat]

# img = ortophoto(path=fp)
# x,y = img.wgs84_to_sweref(coord)

# print(x,y)

# coord=(6573210.52362,161778.12345)

# thr = 100
# coords = [(6573210.52362,161778.12345), (6573210.52362 + thr*2,161778.12345),(6573210.52362+thr,161778.12345+thr),(6573210.52362,161778.12345+thr)]

# img = ortophoto(path=fp)
# img.crop_to_coord(coord=coord)
# area = img.crop_poly(coords=coords)