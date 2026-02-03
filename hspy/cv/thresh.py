# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:31:05 2023

@author: Rehmer
"""

import numpy as np
import matplotlib.pyplot as plt
                      
class Otsu():
    
    def __init__(self,**kwargs):
        """
        q: In Heimann TPArray images Otsu tends to set the treshold to high,
           i.e. pixels representing the edges of an object tend to be 
           classified as background. To counter that effect the parameter q
           can be used to lower the threshold. Instead of taking the pixel
           value that minimises Otsu's criteria as the threshold, the 
           q-th percentile of all calculated criterias is taken to be the 
           threshold.
           As a result, pixels that would have fallen under the threshold
           as background do now pass the lower threshold and are assigned
           to foreground pixels.
        
        """
                
        self.q = kwargs.pop('q',100)
        self.mask = None
        
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self,mask):
        self._mask = mask

    def _moving_average(self,criteria,n=2):
        ret = np.cumsum(criteria, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret[n - 1:] = ret[n - 1:] / n
        
        return ret
        
    def threshold(self,img):
        
        # Otsu thresholding only works for positive pixel intensities
        if np.min(img) < 0:
            offset = abs(np.min(img))
        else:
            offset = 0
        
        # Add offset to make everyting greater equal zero
        img_off = img + offset
        
        # Consider only non nan values 
        img_off = img_off[~np.isnan(img_off)]
        img_off = img_off.flatten()
        
        # testing all thresholds from the minimum to the maximum of the image
        threshold_range = range(int(np.nanmin(img_off)+1),
                                int(np.nanmax(img_off))+1)


        # In all other cases apply Otu's method as usual
        criterias = [self._1d_otsu(img_off, th) for th in threshold_range]

           
        # There might exist mutliple local minima! Espeacially in more
        # noisy images. Search for all local minima
        criterias = np.array(criterias).astype(int)
        
        # Throw out identical criteria values
        idx = ~(criterias[0:-1] == criterias[1::])
        
        threshold_range = np.array(threshold_range)[:-1][idx]
        
        # if at this point threshold_range is empty, do set the threshold :
        # equal to the images minimum, i.e. do not threshold the image at all
        if len(threshold_range) != 0:
        
            criterias = criterias[:-1][idx]
            
            # Smooth critera using moving average
            criterias = self._moving_average(criterias)
            
            
            # Find local minimum on smoothed curve
            loc_min = np.where(((criterias[1::]<=criterias[0:-1])[:-1]) &\
                               ((criterias[0:-1]<=criterias[1::])[1::]))[0]
            
            # If multiple local minima exist, take the furthest to the right
            # Compensate for the index shift in the previous line by adding 1
            if len(loc_min)!=0:
                loc_min = max(loc_min) + 1
            else:
                # If no minimum can be found, set threshold to maximum range value
                # i.e. threshold is highest pixel value in image
                # if len(threshold_range) == 0:
                #     print('debug')
                loc_min = len(threshold_range) - 1
            
            if self.q == 100:
                                # best threshold is the one minimizing the Otsu criteria
                best_threshold = threshold_range[loc_min]
                
            else:
                
                criterias = criterias[0:loc_min+1]
                
                # calculate q-th percentile of calculated minimum
                percentile = -np.percentile(-criterias,self.q)
                
                # Find value closest to that percentile
                best_threshold = \
                    threshold_range[np.argmin(abs(criterias-percentile))]
            
        else:
            best_threshold = int(np.nanmin(img_off))
        
        # Compensate for offset
        best_threshold = best_threshold - offset
        
        # Threshold image
        img_below = img.copy().astype(float)
        img_above = img.copy().astype(float)
        img_below[img_below>=best_threshold] = np.nan
        img_above[img_above<best_threshold] = np.nan
        
        # Set the mask attribute of the thresholder for other classes
        self.mask =  img>=best_threshold
        
        return img_below,img_above
    
    
    
    def _1d_otsu(self,img,th):
        
        # create the thresholded image
        thresholded_img = np.zeros(img.shape)
        thresholded_img[img >= th] = 1

        # compute weights
        nb_pixels = img.size
        nb_pixels1 = np.count_nonzero(thresholded_img)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1

        # if one of the classes is empty, eg all pixels are below or above the
        # threshold, that threshold will not be considered
        # in the search for the best threshold
        if weight1 == 0 or weight0 == 0:
            return np.inf

        # find all pixels belonging to each class
        val_pixels1 = img[thresholded_img == 1]
        val_pixels0 = img[thresholded_img == 0]

        # compute variance of these classes
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

        return weight0 * var0 + weight1 * var1
    
    # def _2d_otsu(self,img,th):
        
    #     # create the thresholded image
    #     thresholded_im = np.zeros(im.shape)
    #     thresholded_im[im >= th] = 1

    #     # compute weights
    #     nb_pixels = im.size
    #     nb_pixels1 = np.count_nonzero(thresholded_im)
    #     weight1 = nb_pixels1 / nb_pixels
    #     weight0 = 1 - weight1

    #     # if one of the classes is empty, eg all pixels are below or above the
    #     # threshold, that threshold will not be considered
    #     # in the search for the best threshold
    #     if weight1 == 0 or weight0 == 0:
    #         return np.inf

    #     # find all pixels belonging to each class
    #     val_pixels1 = im[thresholded_im == 1]
    #     val_pixels0 = im[thresholded_im == 0]

    #     # compute variance of these classes
    #     var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    #     var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    #     return weight0 * var0 + weight1 * var1    
    

    
    
        
        
        
        
        