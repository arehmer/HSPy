import pandas as pd
import numpy as np


from scipy.spatial import distance_matrix

from ..cv.seg import OtsuSeg

class OfflineCVAT():
    
    def __init__(self,w,h,**kwargs):
        self.w = w
        self.h = h
        self.Seg = OtsuSeg(w,h,**kwargs)
    
    def track(self,video):
    
        # Initialize empty dataframe for storing bounding boxes in
        # columns = ['xtl','ytl','xbr','ybr','frame']
        
        
        # video_bboxes = pd.DataFrame(data=[],
        #                       columns = columns)    

        # counter for bounding box for unique identification
        video_bboxes = []
        
        for i in video.index[0:100]:
            
            # Get image
            img = video.loc[i].values.reshape((self.h,self.w))
            
            # Perform segmentation
            img,img_bboxes = self.Seg.segment(img)
            
            # Add frame information
            img_bboxes['frame'] = i
        
            video_bboxes.append(img_bboxes)
        
        # Concatenate to one dataframe
        video_bboxes = pd.concat(video_bboxes)
        
        # Set the frame as index
        video_bboxes = video_bboxes.set_index('frame')
        
        # Provide a unique id for every bounding box
        video_bboxes['bbox_id'] = range(len(video_bboxes))
        
        # Add a column for the outside property
        video_bboxes['outside'] = 0
        
        # Add a column for the keyframe property
        video_bboxes['keyframe'] = 0
        
        # Every bounding box in first frame gets its own track_id
        idx_0 = video_bboxes.index[0]
        video_bboxes.loc[idx_0,'track_id'] = \
            np.arange(0,len(video_bboxes.loc[[idx_0]]),1,dtype=int)
        video_bboxes.loc[idx_0,'keyframe'] = 1
            
        # Go over the whole video again and merge bounding boxes that are 
        # close to each other in subsequent frames to a single track
        
        # Get highest track_id in first frame
        track_id =  int(video_bboxes['track_id'].max())
        
        for i in video_bboxes.index.unique()[1::]:
            
            print(i)
            if i == 81:
                print('debug')
            
            # Get top left corner of all bounding boxes from last and current
            # frame
            bb_last = video_bboxes.loc[[i-1]].copy()
            bb_curr = video_bboxes.loc[[i]].copy()
            
            # Get number of bounding boxes in last and current frame
            num_bb_last = len(bb_last)
            num_bb_curr = len(bb_curr)
            
            # Create a linkage matrix that links box from previous frame
            # to box in current frame
            L = np.zeros((num_bb_last,num_bb_curr))
                        
            # Compute pairwise distances, i.e. a distance matrix
            dist = distance_matrix(bb_last[['xtl','ytl']],
                                   bb_curr[['xtl','ytl']])
            
            # Get minimum either row wise (if more bounding boxes in last frame)
            # or column wise (if more bounding boxes in current frame)
            if num_bb_last<=num_bb_curr:
                
                # Array pointing to rows (boxes from previous frame) that are
                # linked to boces in new frame
                bb_last_rows = list(range(num_bb_last))
                
                # Find row wise minima
                dist_min = np.min(dist,axis=1)
                idx_min = np.argmin(dist,axis=1)
               
                # Check if duplica exist
                s = pd.Series(idx_min)
                dupl = s[s.duplicated()]
                    
                # Loop though duplicates
                for d in dupl:
                    # Find duplica
                    d_idx = np.where(idx_min==d)[0]
                    
                    # Get corresponding distances and find minimum
                    d_idx_min = np.argmin(dist_min[d_idx])
                    
                    # Keep index on entries not corresponding to minimum
                    del_idx = [d for d in d_idx if d !=d_idx[d_idx_min]]
                   
                    # Delete all entries not corresponding to minimum
                    bb_last_rows = np.delete(bb_last_rows,del_idx)
                    idx_min = np.delete(idx_min,del_idx)
                    
                    # old boxes that couldn't be matched get ended
                    bb_last.iloc[del_idx,bb_last.columns.get_loc('outside')] = \
                        1
                    bb_last.iloc[del_idx,bb_last.columns.get_loc('keyframe')] = \
                        1
                    
                # Update linkage matrix
                L[bb_last_rows,idx_min] = 1
                
                # return which current boxes have not been linked yet
                idx_unlinked = list(set(range(num_bb_curr)) - set(idx_min))
                
                # unlinked boxes get new track ids
                new_tracks = np.arange(track_id+1,track_id+1+len(idx_unlinked),1,
                                       dtype=int)
                bb_curr.iloc[idx_unlinked,bb_last.columns.get_loc('track_id')] = \
                    new_tracks
                bb_curr.iloc[idx_unlinked,bb_last.columns.get_loc('keyframe')] = \
                    1
                

                
                # Update the track id
                if len(new_tracks)>0:
                    track_id = max(new_tracks)
                
            else:
                
                # Array pointing to columns (boxes of current frame) that are
                # linked to boxes in previous frame
                bb_curr_cols = list(range(num_bb_curr))
                
                # Find column wise minima
                dist_min = np.min(dist,axis=0)
                idx_min = np.argmin(dist,axis=0)
                
                # Check if duplica exist
                s = pd.Series(idx_min)
                dupl = s[s.duplicated()]
                    
                # Loop though duplicates
                for d in dupl:
                    # Find duplica
                    d_idx = np.where(idx_min==d)[0]
                    
                    # Get corresponding distances and find minimum
                    d_idx_min = np.argmin(dist_min[d_idx])
                    
                    # Keep index on entries not corresponding to minimum
                    del_idx = [d for d in d_idx if d !=d_idx[d_idx_min]]
                   
                    # Delete all entries not corresponding to minimum
                    bb_curr_cols = np.delete(bb_curr_cols,del_idx)
                    idx_min = np.delete(idx_min,del_idx)
                    
                
                L[idx_min,bb_curr_cols] = 1
                
                # return which current boxes have not been linked yet
                idx_unlinked_c = list(set(range(num_bb_curr)) - \
                                            set(bb_curr_cols))
                
                # new boxes that can't be linked to old box get a new 
                # track_id
                new_tracks = np.arange(track_id+1,track_id+1+len(idx_unlinked_c),1,
                                       dtype=int)
                bb_curr.iloc[idx_unlinked_c,bb_last.columns.get_loc('track_id')] = \
                    new_tracks
                bb_curr.iloc[idx_unlinked_c,bb_last.columns.get_loc('keyframe')] = \
                    1
                
                # Update the track id
                if len(new_tracks)>0:
                    track_id = max(new_tracks)
                
                bb_curr.iloc[idx_unlinked_c,
                             bb_last.columns.get_loc('outside')] = 1
                bb_curr.iloc[idx_unlinked_c,
                             bb_last.columns.get_loc('keyframe')] = 1
                
                # Previous tracks that can't be linked to the current frame
                # are ended
                idx_unlinked_p = list(set(range(num_bb_last)) - \
                                            set(idx_min))
                bb_last.iloc[idx_unlinked_p,
                             bb_last.columns.get_loc('outside')] = 1
                bb_last.iloc[idx_unlinked_p,
                             bb_last.columns.get_loc('keyframe')] = 1
                
            # Link remaining boxes to one track according to linkage matrix
            for l in range(num_bb_last):
                link = np.where(L[l]==1)[0]
                bb_curr.iloc[link,bb_curr.columns.get_loc('track_id')] =\
                    bb_last.iloc[l,bb_last.columns.get_loc('track_id')]
            

            # Replace rows in the old video_bboxes with the new frames
            # containing the linkage information
            video_bboxes.loc[[i-1]] = bb_last
            video_bboxes.loc[[i]] = bb_curr
                       
        return video_bboxes

        