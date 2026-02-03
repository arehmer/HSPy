# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:31:05 2023

@author: Rehmer
"""

import os


import numpy as np
import pandas as pd
import pickle as pkl

import matplotlib

import matplotlib.pyplot as plt
from pathlib import Path


from scipy import ndimage

from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.vq import kmeans2

from hspytools.cv.filters import Convolution
from hspytools.cv.thresh import Otsu
from hspytools.clust.gk import GK
# import cv2

    
class Seg():
    
    def __init__(self,w,h,**kwargs):
        
        self.w = w
        self.h = h
        self.pix_frame = kwargs.pop('pix_frame',1)
        self.png_path = kwargs.pop('png_path','seg_png')
        
        
               
        self.lonely_pix_filter = Convolution(mode='same')
        conv_filter = np.zeros((3,3))
        conv_filter[:,1] = 1
        conv_filter[1,:] = 1
        conv_filter[1,1] = 2
        self.lonely_pix_filter.k = conv_filter
    
    def get_proposals(self,**kwargs):
        """
        Takes a single frame as np.ndarray or a whole sequence as pd.DataFrame
        and generates proposals for every frame

        Parameters
        ----------
        video : TYPE
            DESCRIPTION.

        Returns
        -------
        bbox_annnot : TYPE
            DESCRIPTION.

        """
        
        frame = kwargs.pop('frame',None)
        image_id = kwargs.pop('image_id',None)
        video = kwargs.pop('video',None)
        
        
        
        # If a single frame is provided, generate proposals for that one
        if frame is not None:
            
            bbox_frame = self.segment_frame(frame)
            
            # Add a column for the image id
            bbox_frame['image_id'] = image_id
            
            # Name index "id" for compatiblity with all other classes
            bbox_frame.index.rename('id',inplace=True)
            
            return bbox_frame
        
        elif video is not None:
        
            bbox_video = []    
        
            for i in video.index:
                # Convert row from DataFrame to array
                frame = video.loc[i].values.reshape((self.h,self.w))
                
                # Get bounding boxes
                bbox_frame = self.segment_frame(frame)
                
                # Add a column for the image id
                bbox_frame['image_id'] = i
                
                bbox_video.append(bbox_frame)
            
            # Concatenate to one large DataFrame
            bbox_video = pd.concat(bbox_video)
            
            # Reindex so index is unique
            bbox_video = bbox_video.reset_index(drop=True)
            
            # Name index "id" for compatiblity with all other classes
            bbox_video.index.rename('id',inplace=True)
                
            return bbox_video
    
    def bboxes_from_clust(self,pix_coords,clust_dict,**kwargs):
        """
        
        Parameters
        ----------
        pix_coords : array 
            Nx2 array with coordinates of pixels. First column is x-coordinate,
            second column y coordinate
        clust_dict : dict
            Dictionary with structure {cluster_label(int):list/array of indices
                                       of pixels in pix_coords}

        Returns
        -------
        None.

        """
        
        # Initialize empty DataFrame
        columns = ['xtl','ytl','xbr','ybr']
        bboxes = pd.DataFrame(data=[],
                              columns = columns)    
        
        pix_frame = kwargs.pop('pix_frame',self.pix_frame)
        
        for c in clust_dict.keys():
            
            # Get coordinates of all pixels belonging to that cluster
            clust_pix_coords = pix_coords[clust_dict[c],:]
            
            # get the upper left and lower right corner
            x_min,y_min =  clust_pix_coords.min(axis=0)
            x_max,y_max =  clust_pix_coords.max(axis=0)+1
            
            # Add a frame if desired
            x_min = max(0,x_min-pix_frame)
            y_min = max(0,y_min-pix_frame)
            x_max = min(self.w,x_max+pix_frame)
            y_max = min(self.h,y_max+pix_frame)
                       
            
            # Write to dataframe
            bboxes.loc[c] = [x_min,y_min,x_max,y_max] 
            

        
        return bboxes
    
    def filter_lonely_pix(self,xy_coords,pix):
        """
        Filters the selected foregound pixels using a convolution filter to 
        determine the number on nonzero neighbouring pixels and thresholding 
        to get rid of pixels with less than a specified number of neighbours
        
        Parameters
        ----------
        pix : TYPE
            DESCRIPTION.
        pix_coords : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """    
        
        # initialize a zero image
        img_orig = np.zeros((self.h,self.w))
        img_conv = np.zeros((self.h,self.w))
        
        # Restore original image
        img_orig[xy_coords[:,1],xy_coords[:,0]] = pix.flatten()
        
        # Make a binary image for convolution
        img_conv[xy_coords[:,1],xy_coords[:,0]] = 1
        
        # Apply convolution filter
        img_conv = self.lonely_pix_filter.convolve(img_conv)
        
        # Threshold image 
        img_conv[img_conv<3] = 0
        img_conv[img_conv>=3] = 1
        
        # Get coordinates of nonzero pixels
        y_coords_filt,x_coords_filt = np.where(img_conv>0) 
        xy_coords_filt = np.vstack((x_coords_filt,y_coords_filt)).T
        
        # Get pixel intensities
        pix_filt =  img_orig[y_coords_filt,x_coords_filt].reshape((-1,1))
        
        return xy_coords_filt,pix_filt
    
    def write_to_png(self,title,file_name):
        """
        Save the last image and the segmented image as a png for later 
        inspection

        Parameters
        ----------
        title : string
            A string that will be used as the title of the image
        file_name: string
            A string that will be used as the name of the png-file

        Returns
        -------
        None.

        """
        
        img = self.img
        img_seg = self.img_seg
        
        # Use Agg to not show figure
        matplotlib.use('Agg')
        
        # Get maximum and minimum of data for scaling purposes
        vmin = np.nanmin(img)
        vmax = np.nanmax(img)
        
        # Plot as visual control
        plt.ioff()
        fig,ax = plt.subplots(1,2)
        fig.suptitle(title)
        im1 = ax[0].imshow(img,vmin=vmin,vmax=vmax)
        im2 = ax[1].imshow(img_seg,vmin=vmin,vmax=vmax)
    

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)
        
        save_path = Path.cwd() / self.png_path
        
        # Check if directory for saving clustering results exists
        if not save_path.is_dir():
            save_path.mkdir()
        
        # Check if file exists in folder       
        file_path = save_path / file_name
        
        if file_path.exists():
            os.remove(file_path)
        
                
        plt.savefig(file_path,format='png')
        plt.close(fig)
        
        # Change backend to show figures again
        matplotlib.use('Qt5Agg')
        
    
    def _choose_foreground(self,img_below,img_above,dT):
        
        if  dT >= 5:
            img_thresh = img_above  
        else:
            
            foreground_picked = False
            
            while foreground_picked == False:
            
                # In this case the algorithm does not work robustly
                # ask user to decide
                vmin = min(np.nanmin(img_below),np.nanmin(img_above))
                vmax = max(np.nanmax(img_below),np.nanmax(img_above))
                
                fig,ax = plt.subplots(1,2)
                ax[0].imshow(img_below,vmin=vmin,vmax=vmax)
                ax[0].set_title('below threshold')
                im_a = ax[1].imshow(img_above,vmin=vmin,vmax=vmax)
                ax[1].set_title('above threshold')
                fig.colorbar(im_a)
                fig.show()
                plt.pause(10E-2)
                
                print("""User input needed. Choose which image represents the 
                      object.""")
                choice = input("""Enter 'b' for the image below the threshold 
                               or 'a' for the image above the threshold:   """)            
                
                plt.close(fig)
                
                if choice == 'b':
                    img_thresh = img_below
                    foreground_picked = True
                    
                elif choice == 'a':
                    img_thresh = img_above
                    foreground_picked = True
                
        return img_thresh


# class WatershedSeg(Seg):
    
#     def __init__(self,**kwargs):
        
#         init_dict =  kwargs.pop('init_dict',None)
        
#         if init_dict is not None:
#             attr_dict = init_dict
#         else:
#             attr_dict = kwargs
        
        
#         self.w = attr_dict.pop('w',None)
#         self.h = attr_dict.pop('h',None)
        
#         self.bbox_sizelim = attr_dict.pop('bbox_sizelim',{'w_min':5,
#                                                           'w_max':18,
#                                                           'h_min':10,
#                                                           'h_max':28})
        
#         self.dist_thresh = attr_dict.pop('dist_thresh',[0,1,2])
        
        
#         self.thresholder = Otsu(**attr_dict)
        
#         Warning('Remove thresholder and border in future releases!')
#         Warning('Background mean set constant here. Delete background!')
        
#         c = 8
#         self.lap_kernel = np.array([[c/8, c/8, c/8], [c/8, -c, c/8], [c/8, c/8, c/8]],
#                                    dtype=np.float32)
#         self.morph_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
#                                      dtype=np.uint8)
        
#         self.img_seg = {}
        
        
        
#         super().__init__(self.w,self.h,**attr_dict)
        
#     def save(self):
#         '''
#         Returns all non-private an non-builtin attributes of this class
#         as a dictionary with the purpose of reloading this instance from the
#         attribute dictionary. 

#         Returns
#         -------
#         None.

#         '''
        
#         attr_dict = { k:v for k,v in vars(self).items() if not k.startswith('_') }
        
        
#         # save_path = folder / 'WatershedSeg.prop_engine'
#         # pkl.dump(attr_dict,open(save_path,'wb'))
        
#         return attr_dict
    
#     def _threshold_frame(self,img,sharpen):
        
#         # Normalize image
#         # img = np.uint16(img)
#         img_orig = img.copy()
#         # img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        
        
#         # Sharpen image to detect edges better
#         if sharpen==True:
#             img = self._sharpen_img(img)
        
        
#         _,img_above = self.thresholder.threshold(img)
        
#         img_above[~np.isnan(img_above)] = 255 #0
#         img_above[np.isnan(img_above)] = 0 #255
#         img_thresh = np.uint8(img_above)
        

#         return img_thresh
    
#     def _get_foreground(self,img,d_lim):

#         # Use closing on the foreground to carve out the
#         # foreground better
#         img_fg = cv2.morphologyEx(img,cv2.MORPH_OPEN,
#                                   self.morph_kernel,
#                                   iterations = 1)
        
#         # Apply distance transformation
#         img_fg = cv2.distanceTransform(255-img_fg, cv2.DIST_L2, 3)
        
#         # Keep only pixels that are below a certain distance d to the next
#         # nonzero pixel as foreground
#         idx_fg = (img_fg<=d_lim)
        
#         # Set foreground to 255, background to 0
#         img_fg[~idx_fg] = 0
#         img_fg[idx_fg] = 255
        
#         # Convert to proper type
#         img_fg = np.uint8(img_fg)
        
#         return img_fg
    
#     def segment_frame(self,img):
        
#         # Convert image to uint16
#         img = np.uint16(img)
        
#         img_orig = img.copy()
        
#         # Threshold the image, once the original image and once a sharpened 
#         # version
#         img_thresh = self._threshold_frame(img,sharpen=False)
#         img_thresh_sh = self._threshold_frame(img,sharpen=True)
        
#         img_preproc = [img_thresh_sh]#[img_thresh]#,img_thresh_sh]
        
        
#         pix_frame = {0:1,1:0,2:-1,3:-2}
        
#         proposed_boxes = []
        
#         for img_pp in img_preproc:
        
#             for d in self.dist_thresh:
                
#                 # img_fg = img_dist.copy()
#                 img_fg = self._get_foreground(img_pp,d)
                
                
#                 # Get the background by dilating the foreground
#                 img_bg = cv2.dilate(img_pp,
#                                     self.morph_kernel,
#                                     iterations=2)

                
#                 # By calculating the difference between background and foreground
#                 # the unknown space is determined
#                 unknown = cv2.subtract(img_bg,img_fg)
                
#                 # Get markers for watershed algorithm
#                 _, markers = cv2.connectedComponents(img_fg,
#                                                      connectivity=4)
                
#                 # Convert to proper type
#                 markers = markers.astype('int32')
                
#                 markers = markers + 1
                
#                 # Label the unknown area with 0, i.e. to be determined by watershed
#                 markers[unknown==255] = 0
                
#                 # add useless channels to image
#                 img_seg = np.stack([img.copy(),
#                                 np.zeros(img.shape),
#                                 np.zeros(img.shape)],
#                                axis=2)
                
#                 # Apply watershed
#                 img_seg = cv2.watershed(np.uint8(img_seg), markers)
                
#                 # Write to attribute for debugging
#                 self.img_seg[d] = img_seg
                
#                 # Set -1 and 1 to zero. -1 are borders, 1 is the background
#                 img_seg[img_seg==-1] = 0
#                 img_seg[img_seg==1] = 0
                
#                 # Get coordinates of all nonzero pixels
#                 pix_y, pix_x = np.indices(img_seg.shape)
        
#                 pix_xy = np.vstack([pix_x.flatten(),pix_y.flatten()]).T 
                
#                 # create a dictionary mapping storing which pixels belong to which 
#                 # cluster
#                 clust_labels = set(np.unique(img_seg.flatten())) - set([0])
#                 clust_labels = list (clust_labels)
                
#                 clust_dict = {c:img_seg.flatten()==c 
#                               for c in clust_labels}
                
#                 # Create an array with 
#                 bboxes = self.bboxes_from_clust(pix_xy,
#                                                 clust_dict,
#                                                 pix_frame = pix_frame[d])
            
#                 proposed_boxes.append(bboxes)
        
#         # Due to a future warning of pandas, empty dataframes have to be 
#         # deleted before concatenation
#         proposed_boxes = [boxes for boxes in proposed_boxes \
#                           if len(boxes)!=0]
        
#         # Concatenate
#         bboxes = pd.concat(proposed_boxes)
        
#         # Reset index to make it unique
#         bboxes = bboxes.reset_index(drop=True)
        
#         # Calculate the mean of the background for HOG purposes
#         bg_mean = img_orig[img_bg==255].mean()
#         bboxes['bg_mean'] = bg_mean
        
#         # Cast all columns to integers
#         bboxes = bboxes.astype({'xtl':int, 'ytl':int, 'xbr':int,
#                                 'ybr':int,'bg_mean':int})
        
#         # Filter out boxes that are too small
#         bboxes = self._filter_bboxes(bboxes)
        
#         return bboxes
    
#     def _sharpen_img(self,img):
        
#         laplace_img = cv2.filter2D(img, cv2.CV_16U, self.lap_kernel)
#         # sharp_img = np.float32(img)
#         img = img - laplace_img
#         # sharp_img = np.clip(sharp_img, 0, 255)
#         # sharp_img = sharp_img.astype('uint8')
        
#         return img
    
#     def _filter_bboxes(self,bboxes):
#         """
#         Applies customized heuristics for filtering out boundind boxes based on
#         certain criteria.

#         Parameters
#         ----------
#         bboxes : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
        
#         w_min = self.bbox_sizelim['w_min']
#         w_max = self.bbox_sizelim['w_max']
#         h_min = self.bbox_sizelim['h_min']
#         h_max = self.bbox_sizelim['h_max']
 
#         # Filter out boxes that are above or below a certain size 
#         bboxes = bboxes.loc[(bboxes['xbr'] - bboxes['xtl'])>=w_min]
#         bboxes = bboxes.loc[(bboxes['ybr'] - bboxes['ytl'])>=h_min]
#         bboxes = bboxes.loc[(bboxes['xbr'] - bboxes['xtl'])<=w_max]
#         bboxes = bboxes.loc[(bboxes['ybr'] - bboxes['ytl'])<=h_max]        
        
#         return bboxes
  
        
        


class ClustSeg(Seg):
    
    def __init__(self,w,h,**kwargs):
        
        super(ClustSeg,self).__init__(w,h)
  
        self.hierarch_clust = hierarch_clust

    def segment(self,img,**kwargs):

        Ta = kwargs.pop('Ta',np.nan)
        To = kwargs.pop('To',np.nan)        
        
        # Crop the image by 3 pixels to get rid of corners.
        img_crop = img[3:self.h-3,3:self.w-3]

        # Threshold image
        img_below, img_above= threshold(img_crop)
        
        # Depending on difference between ambient and object temperature 
        # the object appears below or above the threshold in the image
        img_thresh = self._choose_foreground(img_below,img_above,To-Ta)
         
        # To preserve shape create an NaN array of the original shape and 
        # insert the thresholded image
        img_temp = np.zeros((self.h,self.w))*np.nan
        img_temp[3:self.h-3,3:self.w-3] = img_thresh
        
        # CLuster the thresholded image
        df_clust,df_clust_stat = self.hierarch_clust(img_temp,
                                                     std_lim=0.01)
        
        # Depending on the relation between ambient temperature Ta and 
        # object temperature the object is represented by the most 
        # or least intensive pixels.
        if  To > Ta:
            c_idx = df_clust_stat['mean_p'].idxmax()  
        elif To < Ta:
            c_idx = df_clust_stat['mean_p'].idxmin()  
        elif To == Ta:
            c_idx = df_clust_stat['mean_d'].idxmin()  
        
        # Select pixels belonging to the cluster c_idx 
        idx_sel = df_clust.loc[df_clust['cluster']==c_idx].index
        
        # Create a new image containing only the selected pixels
        img_seg = np.zeros((self.h,self.w))*np.nan
        img_seg = img_seg.flatten()
        img_seg[idx_sel] = df_clust.loc[idx_sel,'p'].values.flatten()
        
        # reshape
        img_seg = img_seg.reshape((self.h,self.w))

        # Save image and segmented image as attributes
        self.img = img
        self.img_seg = img_seg
        
        return img_seg

class RegionSeg(Seg):
    
    def __init__(self,w,h,**kwargs):
        
        super(RegionSeg,self).__init__(w,h)

        self.r = kwargs.pop('r',3)
  
        # self.hierarch_clust = hierarch_clust
        
    def segment(self,img,**kwargs):
        
        Ta = kwargs.pop('Ta',np.nan)
        To = kwargs.pop('To',np.nan)
        
        # Crop the image by 3 pixels to get rid of corners.
        img_crop = img[3:self.h-3,3:self.w-3]
        
        # Threshold image
        img_below, img_above= Otsu().threshold(img_crop)
        
        # Depending on difference between ambient and object temperature 
        # the object appears below or above the threshold in the image
        img_thresh = self._choose_foreground(img_below,img_above,To-Ta)
        

        # To preserve shape create an NaN array of the original shape and 
        # insert the thresholded image
        img_temp = np.zeros((self.h,self.w))*np.nan
        img_temp[3:self.h-3,3:self.w-3] = img_thresh

        
        # Calculate distance of each pixel from center of gravity
        df_dist = dist_from_center(img_temp)
     
        # Select pixels inside specified radius
        idx_sel = df_dist.loc[(df_dist['d']<=self.r) &\
                              ~(df_dist['p']).isna()].index
        
        # Create a new image array containing only the selected pixels
        img_seg = np.zeros((self.h,self.w))*np.nan
        img_seg = img_seg.flatten()
        img_seg[idx_sel] = df_dist.loc[idx_sel,'p'].values.flatten()
        
        # reshape
        img_seg = img_seg.reshape((self.h,self.w))
        
        # Save image and segmented image as attributes
        self.img = img
        self.img_seg = img_seg
        
        return img_seg



class OtsuSeg(Seg):
    
    def __init__(self,w,h,**kwargs):
        
        super(OtsuSeg,self).__init__(w,h)

        
        self.foreground_lim = kwargs.pop('foreground_lim',0.3)
        
        
        # self.agg_clust = AgglomerativeClustering(n_clusters=None,
        #                                          distance_threshold = distance_threshold,
        #                                          affinity = 'euclidean',
        #                                          linkage = 'ward' )
        
        self.agg_clust = AgglomerativeClustering(n_clusters=None,
                                                 distance_threshold = 2,
                                                 metric = 'manhattan',
                                                 linkage = 'single' )
        
        self.kernel_size =  kwargs.pop('conv_filter_size',3)
        
        # Initialize the convolution filter
        kernel = np.zeros((self.kernel_size,self.kernel_size))
        kernel[self.kernel_size%2,:] = 1
        kernel[:,self.kernel_size%2] = 1
        kernel[self.kernel_size%2,self.kernel_size%2] = 0
        
        self.conv_filter = Convolution(mode='same')
        self.conv_filter.k = kernel
        
        
    def segment(self,img,**kwargs):
        
        # Perform Otsu thresholding on image as many times as specified to
        # obtain the image foreground
        img_seg = img.copy()
        
        foreground_ratio = 1
        
        while foreground_ratio > self.foreground_lim:
            
            _,img_seg = Otsu().threshold(img_seg)
            foreground_ratio = np.sum(~np.isnan(img_seg)) / (self.w*self.h)
            # print(foreground_ratio)
            # plt.figure()
            # plt.imshow(img_seg)
        
        # Replace al NaN with zeros or convolution doesn't work
        img_seg[np.isnan(img_seg)] = 0
        
        # Set all pixels in foreground to 1
        img_seg[img_seg>0] = 1 
        
        # Perform a convolution to get rid of lonely pixels in the foreground
        # that crossed threshold only by chance
        img_seg = self.conv_filter.convolve(img_seg)
        
        # Set all pixels with less than 2 neighbors to zero
        img_seg[img_seg<2] = 0
        
        # Get x- and y- coordinates of all nonzero pixels
        pix_y, pix_x = np.where(img_seg!=0)
        pix_xy = np.vstack([pix_x,pix_y]).T 
        
        # Perform agglomorative cluster to aggregate neigbouring pixels
        # to one large cluster
        
        if pix_xy.shape[0]>=2:
            self.agg_clust.fit(pix_xy)
            
            # Dendrogram for debugging
            # Z = linkage(pix_xy)
            # plt.figure()
            # dendrogram(Z)  
                
            # Get cluster of pixels
            clust_labels = self.agg_clust.labels_   
            
            # Cluster labels start at zero, change that by addition with 1
            clust_labels = clust_labels + 1 
            
            # Assign cluster labels to pixels in image
            img_seg[pix_y,pix_x] = clust_labels
                    
            # Save image and segmented image as attributes
            self.img = img
            self.img_seg = img_seg
        else:
            img_seg = []
            clust_labels = []
        
        return img_seg,clust_labels
        
    
def dist_from_center(img):
    
    data = []
    
    # replace all nan with zeros for this calculation
    img_nonan = np.nan_to_num(img)
    
    center = ndimage.center_of_mass(abs(img_nonan))
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            dist = np.sqrt((x-center[1])**2+(y-center[0])**2)
            data.append([x,y,img[y,x],dist])
    
    df = pd.DataFrame(data = data, columns = ['x','y','p','d'])
    
    
    return df

def hierarch_clust(img,**kwargs):
    '''
    Performs hierarchical clustering on an image given as numpy array
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    dK_std_lim = kwargs.pop('dK_std_lim',0.01)
    dK_lim = kwargs.pop('dK_lim',1)
    
    n_clusters_lim = kwargs.pop('n_clusters_lim',20)
        
    # Calculate distance from center
    df = dist_from_center(img)
   
    # A boolean that controls when to stop
    iterate = True
    
    # Number of clusters to start with
    n_clusters = 3
   
    while iterate == True:
        
        # Perform hierarchical clustering
        hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters,
                                                       affinity='euclidean',
                                                       linkage='ward',
                                                       compute_distances=True)
       
        # Cluster with non nan values
        idx_nonan = df.loc[~df['p'].isna()].index
        
        hierarchical_cluster.fit(df.loc[idx_nonan,['d','p']])
       
        df.loc[idx_nonan,'cluster'] = hierarchical_cluster.labels_ 
        
        # distances in cluster space
        # df.loc[idx_nonan,'clust_dist'] = hierarchical_cluster.distances_
        
        # Evaluate some statistics on found clusters
        df_stat = pd.DataFrame(columns = ['mean_p','std_p','mean_d','std_d']) 
        
        for c in df.loc[idx_nonan,'cluster'].unique():
            mean_p = df.loc[df['cluster']==c,'p'].mean()
            std_p = df.loc[df['cluster']==c,'p'].std()
            
            mean_d = df.loc[df['cluster']==c,'d'].mean()
            std_d = df.loc[df['cluster']==c,'d'].std()
            
            df_stat.loc[int(c)] = [mean_p,std_p,mean_d,std_d]
                        
        # std_rel = abs(df_stat.loc[idx,'std']/df_stat.loc[idx,'mean'])
        dK_std = abs(df_stat['std_p']/df_stat['std_d'])
        dK = df_stat['mean_p'].diff().abs().min()
        
        
        # Do not consider clusters with zero standard deviation (=background)
        dK_std = dK_std.loc[~dK_std.isna()]
        # dK = 
        
        if all(dK_std <= dK_std_lim) or dK < dK_lim or\
            (n_clusters>=n_clusters_lim):
            iterate = False
        else:
            n_clusters = n_clusters + 1
        
        
    return df,df_stat
    

class Kmeans(Seg):
    """
    Class for performing image segmentation using the scipy implementation
    of K-means clustering
    
    https://docs.scipy.org/doc/scipy/reference/cluster.vq.html#module-scipy.cluster.vq
    
    Link to paper explaining the initialization technique:
    https://theory.stanford.edu/%7Esergei/papers/kMeansPP-soda.pdf
    """
    
    def __init__(self,w,h,**kwargs):
        
        
        init_dict =  kwargs.pop('init_dict',None)
        
        if init_dict is not None:
            attr_dict = init_dict
        else:
            attr_dict = kwargs
        
        self.thresholder = Otsu(**kwargs)
        self.k_max = attr_dict.pop('k_max',5) 
        
        self.bbox_sizelim = attr_dict.pop('bbox_sizelim',{'w_min':5,
                                                          'w_max':18,
                                                          'h_min':10,
                                                          'h_max':30})
                
        super().__init__(w,h,**attr_dict)
    
    def segment_frame(self,img):
        
        # Generate a matrix containing cartesian coordinates of pixels in image
        y_coords,x_coords = np.where(~np.isnan(img)) 
        xy_coords = np.vstack((x_coords,y_coords)).T
        
        # Convert pixels in image to rwo vector
        pix = img[xy_coords[:,1],xy_coords[:,0]].reshape((-1,1))
        
        # Perform thresholding using Otsu's method
        pix_bel,pix_abv = self.thresholder.threshold(pix)
        
        # Filter out NaNs, kMeans can't deal with them
        idx_nona = ~np.isnan(pix_abv)
        pix_abv = pix_abv[idx_nona].reshape((-1,1))
        xy_coords_abv = xy_coords[idx_nona.flatten(),:]
        
        # Perform a convolution on the foreground, that gets rid of lonely
        # pixels that just crossed the threshold by chance
        # xy_coords_abv,pix_abv = self.filter_lonely_pix(xy_coords_abv,pix_abv)
        
        # Perform clustering for different numbers of clusters
        
        bboxes = []
        
        for k in range(1,self.k_max):

            clust_dict = self.cluster(pix_abv,xy_coords_abv,k)
        
            # Get Bounding Boxes around pixel clusters
            bboxes_k = self.bboxes_from_clust(xy_coords_abv,clust_dict)
            
            # Append bounding bboxes from this iteration to list
            bboxes.append(bboxes_k)
            
        bboxes = pd.concat(bboxes)    
        
        
        # Delete duplicates
        idx_dupl = bboxes[['xtl','ytl','xbr','ybr']].duplicated()
        bboxes = bboxes.loc[~idx_dupl]
        
        # Calculate the mean of the background for HOG purposes
        bg_mean = np.nanmean(pix_bel)
        bboxes['bg_mean'] = bg_mean
        
        # Cast all columns to integers
        bboxes = bboxes.astype({'xtl':int, 'ytl':int, 'xbr':int,
        'ybr':int,'bg_mean':int})
        
        # Filter out boxes that are too large or too small
        bboxes = self.filter_bboxes(bboxes)
        
        return bboxes

    def filter_bboxes(self,bboxes):
        """
        Applies customized heuristics for filtering out boundind boxes based on
        certain criteria.

        Parameters
        ----------
        bboxes : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        w_min = self.bbox_sizelim['w_min']
        w_max = self.bbox_sizelim['w_max']
        h_min = self.bbox_sizelim['h_min']
        h_max = self.bbox_sizelim['h_max']
 
        # Filter out boxes that are above or below a certain size 
        bboxes = bboxes.loc[(bboxes['xbr'] - bboxes['xtl'])>=w_min]
        bboxes = bboxes.loc[(bboxes['ybr'] - bboxes['ytl'])>=h_min]
        bboxes = bboxes.loc[(bboxes['xbr'] - bboxes['xtl'])<=w_max]
        bboxes = bboxes.loc[(bboxes['ybr'] - bboxes['ytl'])<=h_max]        
        
        return bboxes
        
        
    def cluster(self,pix,pix_coords,k):
        
        # # Normalize pixels and coordinates
        pix_coords_norm = pix_coords / [self.w,self.h]
        
        pix_norm = (pix - pix.min()) / \
            (pix.max(axis=0) - pix.min(axis=0))
        
        # Combine pixel intensities and coordinates to feature space        
        feat = np.hstack((pix_coords_norm,pix_norm))
        # feat = np.hstack((pix_coords,pix))
        
        # feat = pix_coords.astype(float)
        # perform clustering using the k-means++ initialization
        centroid,label = kmeans2(data = feat,
                                 k = k,
                                 minit='++')
        
        # create a dictionary mapping storing which pixels belong to which 
        # cluster
        clust_dict = {c:label==c for c in np.unique(label)}

        return clust_dict
        

class FuzzyGK(Seg):
    """
    Class for performing image segmentation using the scipy implementation
    of K-means clustering
    
    https://docs.scipy.org/doc/scipy/reference/cluster.vq.html#module-scipy.cluster.vq
    
    Link to paper explaining the initialization technique:
    https://theory.stanford.edu/%7Esergei/papers/kMeansPP-soda.pdf
    """
    
    def __init__(self,w,h,**kwargs):
        
        self.thresholder = Otsu(**kwargs)
        self.k_max = 5
        
        self.bbox_sizelim = kwargs.pop('bbox_sizelim',(4,30)) 
        
        super().__init__(w,h,**kwargs)
    
    def segment_frame(self,img):
        
        # Generate a matrix containing cartesian coordinates of pixels in image
        y_coords,x_coords = np.where(~np.isnan(img)) 
        xy_coords = np.vstack((x_coords,y_coords)).T
        
        # Convert pixels in image to rwo vector
        pix = img[xy_coords[:,1],xy_coords[:,0]].reshape((-1,1))
        
        # Perform thresholding using Otsu's method
        pix_bel,pix_abv = self.thresholder.threshold(pix)
        
        # Filter out NaNs, kMeans can't deal with them
        idx_nona = ~np.isnan(pix_abv)
        pix_abv = pix_abv[idx_nona].reshape((-1,1))
        xy_coords_abv = xy_coords[idx_nona.flatten(),:]
        
        # Perform a convolution on the foreground, that gets rid of lonely
        # pixels that just crossed the threshold by chance
        # xy_coords_abv,pix_abv = self.filter_lonely_pix(xy_coords_abv,pix_abv)
        
        # Perform clustering for different numbers of clusters
        
        bboxes = []

        for k in range(2,self.k_max):

            clust_dict = self.cluster(pix_abv,xy_coords_abv,k)
        
            # Get Bounding Boxes around pixel clusters
            bboxes_k = self.bboxes_from_clust(xy_coords_abv,clust_dict)
            
            # Append bounding bboxes from this iteration to list
            bboxes.append(bboxes_k)
            
        bboxes = pd.concat(bboxes)    
        
        
        # Delete duplicates
        idx_dupl = bboxes[['xtl','ytl','xbr','ybr']].duplicated()
        bboxes = bboxes.loc[~idx_dupl]
        
        # Calculate the mean of the background for HOG purposes
        bg_mean = np.nanmean(pix_bel)
        bboxes['bg_mean'] = bg_mean
        
        # Cast all columns to integers
        bboxes = bboxes.astype({'xtl':int, 'ytl':int, 'xbr':int,
        'ybr':int,'bg_mean':int})
        
        # Filter out boxes that are too large or too small
        bboxes = self.filter_bboxes(bboxes)
        
        return bboxes

    def filter_bboxes(self,bboxes):
        """
        Applies customized heuristics for filtering out boundind boxes based on
        certain criteria.

        Parameters
        ----------
        bboxes : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        min_size = self.bbox_sizelim[0]
        max_size = self.bbox_sizelim[1]
        
        # Filter out boxes that are above or below a certain size 
        bboxes = bboxes.loc[(bboxes['xbr'] - bboxes['xtl'])>=min_size]
        bboxes = bboxes.loc[(bboxes['ybr'] - bboxes['ytl'])>=min_size]
        bboxes = bboxes.loc[(bboxes['xbr'] - bboxes['xtl'])<=max_size]
        bboxes = bboxes.loc[(bboxes['ybr'] - bboxes['ytl'])<=max_size]        
        
        return bboxes
        
        
    def cluster(self,pix,pix_coords,k):
        
        # # Normalize pixels and coordinates
        pix_coords_norm = pix_coords / [self.w,self.h]
        
        pix_norm = (pix - pix.min()) / \
            (pix.max(axis=0) - pix.min(axis=0))
        
        # Combine pixel intensities and coordinates to feature space        
        feat = np.hstack((pix_coords_norm,pix_norm))
        
        # feat = pix_coords_norm
        
        # Initialize Fuzzy GK Clustering 
        fuzzyGK = GK(n_clusters = k)
        
        # Fit 
        fuzzyGK.fit(feat)
        
        # Get labels of pixels
        label = fuzzyGK.predict(feat)
        
        # create a dictionary mapping storing which pixels belong to which 
        # cluster
        clust_dict = {c:label==c for c in np.unique(label)}

        return clust_dict        
        

class SelectiveSearch(Seg):
    
    """ 
    Class for performing image segmentation using the scipy implementation
    of agglomoroative clustering
    
    Linkage documentation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    """
    
    def __init__(self,w,h,**kwargs):
        
        init_dict =  kwargs.pop('init_dict',None)
        
        if init_dict is not None:
            attr_dict = init_dict
        else:
            attr_dict = kwargs

        self.bbox_lim = attr_dict.pop('bbox_lim',(0,np.inf))
        self.hierarch_clust = linkage
        
        self.thresholder = Otsu(**kwargs)

        
        self.linkage_method = attr_dict.pop('linkage_method','weighted')
        self.linkage_metric = attr_dict.pop('linkage_metric','euclidean')
        
        self.bbox_sizelim = attr_dict.pop('bbox_sizelim',{'w_min':5,
                                                          'w_max':18,
                                                          'h_min':10,
                                                          'h_max':30})
        
        super().__init__(w,h,**attr_dict)
        
        # Test if segmentation works with ths method-metric combination
        # Use a random image with uniformly distributed values
        # between 0 and 4000 dK 
        self.segment_frame(np.random.rand(h,w)*4*1E3)
        
    def _distance_metric(self,X):
        """
        Pairwise distances between observations in n-dimensional space
        """
        pass
        
    
    # def get_proposals(self,video):
        
    #     bbox_annnot = []
        
    #     for i in video.index:
    #         # Convert row from DataFrame to array
    #         frame = video.loc[i].values.reshape((self.h,self.w))
            
    #         # Get bounding boxes
    #         bbox_frame = self.segment_frame(frame)
            
    #         # Add a column for the image id
    #         bbox_frame['image_id'] = i
            
    #         bbox_annnot.append(bbox_frame)
        
    #     # Concatenate to one large DataFrame
    #     bbox_annnot = pd.concat(bbox_annnot)
        
    #     # Reindex so index is unique
    #     bbox_annnot = bbox_annnot.reset_index(drop=True)
        
    #     # Name index "id" for compatiblity with all other classes
    #     bbox_annnot.index.rename('id',inplace=True)
            
    #     return bbox_annnot
    
        
    def segment_frame(self,img):
        
        y_coords,x_coords = np.where(~np.isnan(img)) 
        xy_coords = np.vstack((x_coords,y_coords)).T
        
        # Get pixels and their coordinates
        # xy_coords = xy_coords[idx_sel,:]
        pix = img[xy_coords[:,1],xy_coords[:,0]].reshape((-1,1))
        
        # Perform Otsu-Thresholding here
        below_thresh,above_thresh = self.thresholder.threshold(pix)
        
        # plt.figure()
        # plt.imshow(above_thresh.reshape((40,60)))
        
        # Calculate the mean of the background for HOG purposes
        bg_mean = np.nanmean(below_thresh)
        
        # Get only pixels above threshold
        idx_sel = ~np.isnan(above_thresh)
        pix = above_thresh[idx_sel].reshape((-1,1))
        xy_coords = xy_coords[idx_sel.flatten(),:]
        
        # plot image for debugging
        img_debug = np.zeros(img.shape)
        img_debug[xy_coords[:,1],xy_coords[:,0]] = pix.flatten()

        # plt.figure()
        # plt.imshow(img_debug)
        
        # Cluster in that space
        feat_space = np.hstack((pix,xy_coords))
        # feat_space = xy_coords
        feat_space = (feat_space - feat_space.min(axis=0)) / \
            (feat_space.max(axis=0) - feat_space.min(axis=0))
        # Normalize all values to the same range
        
        
        # feat_space = xy_coords
        
        Z = self.hierarch_clust(feat_space,
                                method=self.linkage_method,
                                metric = self.linkage_metric) 
        
        # Get cluster labels for leaf pixels on all scales within the specified
        # cluster size self.clust_lim
        clust_dict = self._cluster_from_linkage(Z)
        
        
        # Extract Bounding Boxes for all clusters that are within a specified
        # range regarding the number of pixels they contain
        bboxes = self.bboxes_from_clust(xy_coords, clust_dict)
        
        bboxes['bg_mean'] = bg_mean
        
        # Cast columns with corners to integers
        bboxes = bboxes.astype({'xtl':int, 'ytl':int, 'xbr':int,
        'ybr':int,'bg_mean':int})
        
        # Sort by fill ratio
        # bboxes = bboxes.sort_values('fill_ratio',axis=0,ascending=False)
        
        # Delete duplicates
        idx_dupl = bboxes[['xtl','ytl','xbr','ybr']].duplicated()
        bboxes = bboxes.loc[~idx_dupl]
        
        return bboxes
        
    def _cluster_from_linkage(self,Z):
        """
        Assigns orginal observations to clusters according to the provided
        linkage matrix Z
        
        Parameters
        ----------
        Z : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """

        # First find all clusters within the specified size limits
        clust_idx = np.where((Z[:,3] >= self.bbox_lim[0]) &\
                             (Z[:,3] <= self.bbox_lim[1]))[0]
        
        # The number of original observations is the number of rows in the
        # linkage matrix + 1
        N = Z.shape[0]+1
        # For each of the clusters within the size limits, find all their leaves
        # i.e. the corresponding pixels
        
        
        clust = {}
        
        for c in clust_idx:
            
            pix = []
            branch = []
            
            # get the c-th row of the linkage matrix
            z = Z[c,0:2].astype(int)
            
            branch.extend(list(z.astype(int)))
            
            # propagate down until all leafes have been found
            for b in branch:
                if b<N:
                   pix.append(b) 
                else:
                    z = Z[b-N,0:2].astype(int)
                    branch.extend(list(z.astype(int)))
                    
            clust[c] = pix
            
        return clust
            
                        
 