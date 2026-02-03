# import cv2 as cv
import numpy as np
from scipy.signal import correlate
from scipy.special import iv
# import pandas as pd

class Gradient():
    
    def __init__(self,**kwargs):
        
        gradient_method = kwargs.pop('gradient_method','simple')
        
        if gradient_method == 'sobel':
            self._gradient_method = self.gradient_sobel        
        elif gradient_method == 'simple':
            self._gradient_method = self.gradient_simple 

    def _convolve(self,img,k):
        G = correlate(img,k,'valid')
        
        return G
    
    def calculate(self,img):
        return self._gradient_method(img)

    def gradient_sobel(self,img):
        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        ky = np.array([[-1,-2,-1],[0,0,0],[+1,+2,+1]])
            
        Gx = self._convolve(img,kx)
        Gy = self._convolve(img,ky)
        
        M = np.sqrt(np.power(Gx,2)+np.power(Gy,2))
        O = np.arctan2(Gy,Gx)
        
        return M,O    
    
    
    def gradient_simple(self,img):
        
        kx = np.array([[0,0,0],[-1,0,1],[0,0,0]])
        ky = np.array([[0,-1,0],[0,0,0],[0,1,0]])
        
        Gx = self._convolve(img,kx)
        Gy = self._convolve(img,ky)
        
        M = np.sqrt(np.power(Gx,2)+np.power(Gy,2))
        O = np.arctan2(Gy,Gx)
        
        return M,O


class Convolution():
    
    def __init__(self,**kwargs):
        
        kernel = kwargs.pop('kernel',None)
        self.mode = kwargs.pop('mode','valid')
        
        if kernel == 'gaussian':
                       
            self.k = self.kernel_gaussian(**kwargs)        
        
        else:
            self.k = np.zeros((3,3))
            print('Set a custom kernel by setting the attribute "k".')
    def convolve(self,img):
    
        G = correlate(img,self.k,self.mode)
        
        return G
    
    
    def kernel_gaussian(self,**kwargs):
        
        self.size = kwargs.pop('size',3)
        self.sigma = kwargs.pop('sigma',1)
        
        if self.size%2 != 1:
            print("size must be an uneven number!")
            return None
        if self.sigma <= 0:
            print("sigma must be a positive number! ")
            return None
        
        k = np.zeros((self.size,self.size))
        
        c = 1/(2*np.pi*self.sigma**2)
        
        # Coordinates in x and y direction from the center
        x_coord = np.arange(0,self.size,1).reshape(1,-1)-(self.size-1)/2
        y_coord = np.arange(0,self.size,1).reshape(-1,1)-(self.size-1)/2
        
        # Calculate euclidean distance from center       
        D = (np.sqrt(np.tile(x_coord,(self.size,1))**2  +  
                    np.tile(y_coord,(1,self.size))**2) )
        
        # A discrete gaussian kernel is calculated 
        # https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
        k = np.exp(-self.sigma) * iv(D,self.sigma)
        
        # Normalize kernel to avoid altering brightness
        k = k/sum(sum(k))
    
        return k
        
        
# # -*- coding: utf-8 -*-
# """
# Created on Fri Mar 24 10:28:10 2023

# @author: Rehmer
# """

# import numpy as np

# class AWA():
    
#     """
#     Spatio-temporal filter published in Adaptive motion-compensated filtering 
#     of noisy image sequences (Ozkan, 1993)
#     """
    
#     def __init__(self,width,height,**kwargs):
#         """
        

#         Parameters
#         ----------
#         width: int
#             Width of picture in pixels
#         height: int
#             Height of picture in pixels
#         **kwargs : dict
#             DESCRIPTION.
#         K: int
#             Defines order of temporal support of the filter
#         S: int
#             Defines spatial support of the filter as a frame with size S around
#             the pixel to be filtered
#         a: int
#             Parameter for filter weight
#         eps: int
#             Threshold parameter for filter. Reccomendation from paper: set eps**2 
#                 to "two times the value of the noise variance" 
#         Returns
#         -------
#         None.

#         """
        
#         self.width = width
#         self.height = height
#         self.K = kwargs.pop('K',1)
#         self.S = kwargs.pop('S',1)

#         self.a = kwargs.pop('a',1)
#         self.eps = kwargs.pop('eps',1)
        
        
#         # 3-dimensional array that keeps images in storage for filter operations
#         self.img_buffer = np.zeros((self.height,self.width,2*self.K+1))
#         self.idx_buffer = np.zeros((1,1,2*self.K+1))
#         self.buffer_size = 0
#         # Pre calculate indices of support pixels
#         self.__calc_support()
        
#     def reset_buffer(self):
#         """
#         Reset buffer
#         """
#         self.buffer_size = 0
#         self.img_buffer = np.zeros((self.height,self.width,2*self.K+1))
#         self.idx_buffer = np.zeros((1,1,2*self.K+1))
        
#     def __calc_support(self):
#         """
#         Calculates the spatial and temporal support (=indices) for each pixel
#         """
        
#         # Calculate all possible spatial coordinates (upper left = [0,0],
#         # lower right = [self.height,self.width])
#         x_coords = np.arange(0,self.width,1)
#         y_coords = np.arange(0,self.height,1)
        
#         # Reshape such that first column is vertical (y) coordinate and 
#         # second column is horizontal (x) coordinate. Order is such that 
#         # picture is looped over row by row from left to right
#         pixel_coords = np.array(np.meshgrid(y_coords, x_coords))
#         pixel_coords = pixel_coords.T.reshape((-1,2))

#         # Calculate spatial support for each pixel, m contains all possible  
#         # offsets in x and y direction from current pixel 
#         m_range =  np.arange(-self.S,self.S+1,1)
#         m = np.array(np.meshgrid(m_range, m_range))
#         m = m.reshape((2,-1)).T
        
#         # Loop over all pixel coordinates and calculate coordinates of 
#         # neighbouring pixels that belong to the pixels filter support
        
#         # Initialize dictionary to save coordinates for filter support in
#         # (Make it a matrix later for speed up)
#         spat_supp = {tuple(pixel_coords[p,:]):np.zeros(m.shape) 
#                      for p in range(pixel_coords.shape[0])}
        
#         # Each pixel coordinate is a key in the dictionary, the value is an 
#         # array with 2 columns, the first containing the y-coordinate and
#         # the second the x-coordinate of the supporting pixels
#         for p in spat_supp.keys():
            
#             supp = p - m
            
#             # Delete any coordinates outside the image borders
#             supp = supp[np.all(supp>=0,axis=1)] 
#             supp = supp[np.all(supp < [self.height,self.width],axis=1)]
            
#             # The spatial support does not change over time in this implementation,
#             # i.e. the temporal support can be calculated by replicating 
#             # the spatial coordinates over the number of frames in the buffer
            
#             frame_idx = np.arange(0,2*self.K+1,1)
#             frame_idx = np.repeat(frame_idx,len(supp),axis=0).reshape((-1,1))
            
#             supp = np.tile(supp,(2*self.K+1,1))
#             supp = np.hstack([supp,frame_idx])
            
#             # Finally write spatial and temporal indices in dictionary
#             spat_supp[p] = supp
        
#         self.spat_supp = spat_supp
        
#         return None

#     # def filter_img(self,img):
        
#     #     # Update buffer
#     #     # self.__update_buffer(img)
        
#     #     # Update filter coefficients
#     #     self.__update_filter_coeff(img)

#     def update_buffer(self,img,idx):
#         """
#         Updates the image buffer using by adding a new image and deleting the
#         oldest

#         Parameters
#         ----------
#         img : array
#             array containing a new image to be added to the buffer

#         Returns
#         -------
#         None.

#         """
        
#         # Add image to buffer
#         img_buffer = np.dstack([self.img_buffer,img])
#         self.img_buffer = img_buffer[:,:,1::]
        
#         # Add index of image to buffer
#         idx_buffer = np.dstack([self.idx_buffer,idx])
#         self.idx_buffer = idx_buffer[:,:,1::]
        
        
#         # check if buffer is already full, otherwise increase buffer_size
#         if self.buffer_size < 2*self.K+1:
#             self.buffer_size = self.buffer_size +1
        
#         if self.buffer_size == 2*self.K+1:
#             buffer_full = True
#         else:
#             buffer_full = False
        
#         return buffer_full
    
#     def filter_img(self,img,idx):
#         """
#         Calculates the adaptive filter coefficients for each pixel. Right now
#         the calculation is done by looping over the dictionary that contains 
#         the spatial and temporal support indices for each pixel. This comp-
#         utation might be sped up by vectorization

#         Parameters
#         ----------
#         XXX : xxx
#             xxxxxxxxx

#         Returns
#         -------
#         None.

#         """
        
#         buffer_full = self.update_buffer(img,idx)
        
#         if buffer_full == True:
        
#             img = self.img_buffer[:,:,self.K]
#             frame = self.idx_buffer[:,:,self.K]
            
#             filt_img = np.zeros((self.height,self.width))
            
#             for p in self.spat_supp.keys():
                
#                 # Calculate difference in intensity of image w.r.t to all support
#                 # pixels
#                 supp = self.spat_supp[p]
#                 supp_pix = self.img_buffer[supp[:,0],supp[:,1],supp[:,2]]
#                 diff = (img[p]-supp_pix)
            
#                 diff = self.a*np.maximum(self.eps**2,diff**2)
                
#                 # Calculate normalization constant
                     
#                 h = 1/(1+diff)
                
#                 h = h/sum(h)
                
#                 f_est =  h.dot(supp_pix)
                
#                 filt_img[p] = f_est
            
#         else:
#             filt_img = img
#             frame = idx
            
#         return filt_img,int(frame)
    
#     def filter_sequence(self,video):
#         """
#         Filters a whole video sequence given as pandas DataFrame
        

#         Parameters
#         ----------
#         video : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
#         # Reset buffer        
#         self.reset_buffer()

#         # Initialize list for filtered images
#         filt_imgs = []
#         frames = []

#         # Loop over video sequence
#         for i in video.index:
#             img = video.loc[i].values.reshape(self.height,self.width)
#             filt_img,frame =  self.filter_img(img,i)
#             filt_img =  filt_img.flatten()
            
#             filt_imgs.append(filt_img)
#             frames.append(frame)
        
#         # Convert to DataFrame
#         filt_video = pd.DataFrame(data = filt_imgs,
#                                   columns = video.columns,
#                                   index = frames)

#         # The remaining frames in the buffer that have not been filtered are
#         # appended so the overall number of frames remains unchanged
#         filt_video = pd.concat([filt_video,video.iloc[-self.K::]])
        
#         # Delete frames that appear twice (unfiltered and filtered)
#         filt_video = filt_video[~filt_video.index.duplicated(keep='last')]
        
#         return filt_video

# class Bilateral_CV():
    
#     def __init__(self,**kwargs):
        
#         self.d = int(kwargs.pop('d',3))
#         self.sigmaColor = int(self.d**2)
#         self.sigmaSpace =  int(self.d/2)
    
#     def filter_img(self,img,idx):
        
#         img = img.astype(np.float32)
        
#         filt_img = cv.bilateralFilter(img,
#                                       self.d,
#                                       self.sigmaColor,
#                                       self.sigmaSpace)
        
#         return filt_img,idx
    
# class AWA_CV():
    
#     """
#     Spatio-temporal filter published in Adaptive motion-compensated filtering 
#     of noisy image sequences (Ozkan, 1993)
#     """
    
#     def __init__(self,width,height,**kwargs):
#         """
        

#         Parameters
#         ----------
#         width: int
#             Width of picture in pixels
#         height: int
#             Height of picture in pixels
#         **kwargs : dict
#             DESCRIPTION.
#         K: int
#             Defines order of temporal support of the filter
#         S: int
#             Defines spatial support of the filter as a frame with size S around
#             the pixel to be filtered
#         a: int
#             Parameter for filter weight
#         eps: int
#             Threshold parameter for filter. Reccomendation from paper: set eps**2 
#                 to "two times the value of the noise variance" 
#         Returns
#         -------
#         None.

#         """
        
#         self.width = width
#         self.height = height
#         self.K = kwargs.pop('K',1)
#         self.S = kwargs.pop('S',1)

#         self.a = kwargs.pop('a',1)
#         self.eps = kwargs.pop('eps',1)
    
#         self.initialize_filter()
    
#     def initialize_filter(self):
#         # Initialize C++ implementation of filter
#         self.awa_filter = cv.htpa.AWA_Filter(self.width,
#                                              self.height, 
#                                              self.K, 
#                                              self.S, 
#                                              self.a, 
#                                              self.eps)
        
#     def filter_img(self,img,idx):
#         """
#         Calculates the adaptive filter coefficients for each pixel. Right now
#         the calculation is done by looping over the dictionary that contains 
#         the spatial and temporal support indices for each pixel. This comp-
#         utation might be sped up by vectorization

#         Parameters
#         ----------
#         XXX : xxx
#             xxxxxxxxx

#         Returns
#         -------
#         None.

#         """
        
#         img = img.astype(np.float32)
#         filt_img = np.zeros(img.shape)
        
#         frame,filt_img = self.awa_filter.filter(img,int(idx),filt_img)
#         # buffer=self.awa_filter.img_buffer
#         return filt_img,int(frame)
    
#     def filter_sequence(self,video):
#         """
#         Filters a whole video sequence given as pandas DataFrame
        

#         Parameters
#         ----------
#         video : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
#         # Reset buffer        
#         self.initialize_filter()

#         # Initialize list for filtered images
#         filt_imgs = []
#         frames = []

#         # Loop over video sequence
#         for i in video.index:
#             img = video.loc[i].values.reshape(self.height,self.width)
#             filt_img,frame =  self.filter_img(img,i)
#             filt_img =  filt_img.flatten()
            
#             filt_imgs.append(filt_img)
#             frames.append(frame)
        
#         # Convert to DataFrame
#         filt_video = pd.DataFrame(data = filt_imgs,
#                                   columns = video.columns,
#                                   index = frames)

#         # The remaining frames in the buffer that have not been filtered are
#         # appended so the overall number of frames remains unchanged
#         filt_video = pd.concat([filt_video,video.iloc[-self.K::]])
        
#         # Delete frames that appear twice (unfiltered and filtered)
#         filt_video = filt_video[~filt_video.index.duplicated(keep='last')]
        
#         return filt_video
        
# class Scale:
    
#     def __init__(self,**kwargs):
        
#         self.scale = kwargs.pop('scale',(8,10))
                
#         self.GaussianFilter = Convolution('gaussian',**kwargs)
    
#     def set_filter_kernel(self,**kwargs):
        
#         self.GaussianFilter = Convolution('gaussian',**kwargs)
        
#     def _filter(self,img):
        
#         return self.GaussianFilter.convolve(img)
        
#     def downscale(self,img):
        
#         if img is None:
#             return None
        
#         # Filter image first to avoid aliasing
#         filt_img = self._filter(img)
        
#         # To downsample an image by a factor of 4/5 it is first upsampled
#         # by a factor of 4 and then downsampled by a factor of 5
#         up = self.scale[0]
#         down = self.scale[1]
        
#         # Replicate in both directions
#         repl_pic = np.repeat(filt_img, repeats=up, axis=0)
#         repl_pic = np.repeat(repl_pic, repeats=up, axis=1)
        
#         down_img = repl_pic[0::down,0::down]
        
        
#         return down_img