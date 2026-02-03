import numpy as np
import cv2

from ..cv.filters import Scale

class Gaussian_old():
    
    def __init__(self,width,height,**kwargs):
        """
        Class for calculating the pyramid representation of an image
    

        Parameters
        ----------
        width : int
            width of the original image
        height : int
            height of the original image
        **kwargs : dict
            see __init__() for possible keyword arguments

        Returns
        -------
        None.

        """
        
        self.scale = kwargs.pop('scale',(9,10))
        self.num_levels = kwargs.pop('num_levels',4)
        
        # Pre-caclulate the expected size of the image on every level
        self.level_scale = ((1,1),)
        self.level_size = ((width,height),)
        
        for l in range(1,self.num_levels):    
            level_scale = (int((self.scale[0])**l),int((self.scale[1])**l))
            
            level_size = ((width-2)*(level_scale[0]/level_scale[1]), \
                             (height-2)*(level_scale[0]/level_scale[1]))
            level_size = (int(np.ceil(level_size[0])),int(np.ceil(level_size[1])))
            
            self.level_scale = (*self.level_scale,level_scale)
            self.level_size = (*self.level_size,level_size)
            
        # To scale image to the desired size, an instance of a scaler is 
        # needed for each level of the pyramid
        self.ImageScalers = ()

        for l in range(1,self.num_levels):    
            scaler = Scale(scale= self.level_scale[l])
            
            self.ImageScalers = (*self.ImageScalers,scaler)
            
    def construct_pyramid(self,img,**kwargs):
        """
        Take the image, return a dictionary with pyramid representation 
        of the image
        """
        pyramid = {}
        
        thresh_img = kwargs.pop('thresh_img',None)
        
        pyramid[0]  = {}
        
        pyramid[0]['img']= img
        pyramid[0]['thresh_img']= thresh_img
        
        
        for l in range(1,self.num_levels):
            pyramid[l]  = {}
            
            pyramid[l]['img'] = self.ImageScalers[l-1].downscale(img)
            
            pyramid[l]['thresh_img'] = self.ImageScalers[l-1].downscale(thresh_img)
        
        self.pyramid = pyramid
        
        return pyramid
    
    def plot_pyramid(self):
        print('Code for plotting all levels of pyramid of an image')
        
class Gaussian():
    
    def __init__(self,width,height,**kwargs):
        """
        Class for calculating the pyramid representation of an image
    

        Parameters
        ----------
        width : int
            width of the original image
        height : int
            height of the original image
        **kwargs : dict
            see __init__() for possible keyword arguments

        Returns
        -------
        None.

        """
        
        self.scale = kwargs.pop('scale',1.2)
        self.levels = kwargs.pop('levels',3)
        
        # Pre-caclulate the expected size of the image on every level
        # self.level_scale = ((1,1),)
        self.level_size = {0:(width,height)}
        
        for l in range(1,self.levels):    
            scale_level = self.scale**l
            
            w_level = int(np.ceil(width / scale_level))
            h_level = int(np.ceil(height / scale_level))
            level_size = (w_level,h_level)

            self.level_size[l] = (level_size)
            
            
    def construct_pyramid(self,img,**kwargs):
        """
        Take the image, return a dictionary with pyramid representation 
        of the image
        """
        
        
        
        pyramid = {}
        
        thresh_img = kwargs.pop('thresh_img',None)
        
        pyramid[0]  = {}
        
        pyramid[0]['img']= img
        pyramid[0]['thresh_img']= thresh_img
        
        
        for l in range(1,self.levels):
            pyramid[l]  = {}
            
            pyramid[l]['img'] = cv2.resize(img.astype(np.int16), 
                                           dsize=self.level_size[l])
            
            # pyramid[l]['thresh_img'] = self.ImageScalers[l-1].downscale(thresh_img)
        
        self.pyramid = pyramid
        
        return pyramid
    
    def plot_pyramid(self):
        print('Code for plotting all levels of pyramid of an image')