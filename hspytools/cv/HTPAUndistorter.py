import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

class HTPA_Undistorter:
    
    def __init__(self, w, h, pixpitch):
        
        self.w = w
        self.h = h
        self.pixpitch = pixpitch
        self.GridDistortion = None
        
    @property
    def pixpitch(self):
        return self._pixpitch
    @pixpitch.setter
    def pixpitch(self,pixpitch:float):
        self._pixpitch = pixpitch
        print(f'Pixel pitch set to {pixpitch} mm.')
        
    @property
    def w(self):
        return self._w
    @w.setter
    def w(self,w):
        self._w = w
        print(f'Width of thermopile array: {w} pixel.')
    @property
    def h(self):
        return self._h    
    @h.setter
    def h(self,h):
        self._h = h
        print(f'Height of thermopile array: {h} pixel.')


    @property
    def GridDistortion(self):
        return self._GridDistortion
    @GridDistortion.setter
    def GridDistortion(self,GridDistortion:pd.DataFrame):
        self._GridDistortion = GridDistortion
        
    def import_GridDistortionData(self,
                                  GridDistortionData_txt : Path):
        
        if not isinstance(GridDistortionData_txt,Path):
            raise TypeError('GridDistortionData_txt needs to be a Path object!')
        
        
        # Import txt file
        GridDistortionData = pd.read_csv(GridDistortionData_txt,
                                         encoding = 'utf-16-le',
                                         skiprows=12,
                                         delimiter='\t')
                                         
        # Delete NaN rows which stem from Zemax' footer
        GridDistortionData = GridDistortionData.loc[~GridDistortionData.isna().any(axis=1)]
        
        # Delete unnecessary blanks from columns headers
        columns_stripped = []
        for col in GridDistortionData.columns:
            columns_stripped.append(col.strip())
        
        GridDistortionData.columns = columns_stripped
        
        # Check if all exoected columns exist
        expected_columns = ['i', 'j', 'X-Field', 'Y-Field', 'R-Field',
                            'Predicted X', 'Predicted Y', 'Real X', 'Real Y',
                            'Distortion']
        
        for col in expected_columns:
            if not col in GridDistortionData.columns:
                raise ValueError(f'Column {col} missing in GridDistortionData!')

        # Delete the Distortion column because that % sign is unnecessary trouble
        GridDistortionData = GridDistortionData.drop(columns = ['Distortion'])

        # Convert to proper data types
        dtypes = {'i':int,
                  'j':int,
                  'X-Field':float,
                  'Y-Field':float,
                  'R-Field':float,
                  'Predicted X':float,
                  'Predicted Y':float,
                  'Real X':float,
                  'Real Y':float}
        
        GridDistortionData  = GridDistortionData.astype(dtypes)
        
        # If everything is fine, assign imported data to attribute        
        self.GridDistortion = GridDistortionData
        
        # Plot the grid distortion data as well
        self.plot_GridDistortionData()
        
    def plot_GridDistortionData(self):
        
        data = self.GridDistortion

        # Get colormap
        tab10 = plt.get_cmap('tab10')
        
        # Get first two colors
        c0 = tab10(0)   # blue
        c1 = tab10(1)   # orange
        c3 = tab10(3)   # red
        
        # Create figure
        fig, ax = plt.subplots(1,1,figsize = (10,6))
        
        # Plot ideal/undistorted pixel positions
        data.plot.scatter(x = 'Predicted X',
                          y = 'Predicted Y',
                          ax = ax,
                          grid = True,
                          marker = 'o',
                          edgecolors = c0,
                          facecolors = 'none',
                          c='none')

        # Plot distorted pixel positions        
        data.plot.scatter(x = 'Real X',
                          y = 'Real Y',
                          ax = ax,
                          grid = True,
                          marker = 'x',
                          color=c1)
        
        # Plot dimension of thermopile array as rectangle
        w_tp = self.w * self.pixpitch
        h_tp = self.h * self.pixpitch
        
        xtl = -(1/2 + self.w/2) * self.pixpitch
        ytl = -(1/2 + self.h/2) * self.pixpitch
        
        tp_array = matplotlib.patches.Rectangle((xtl,ytl),
                                                w_tp,
                                                h_tp,
                                                lw = 2,
                                                ls = '--',
                                                fc = 'none',
                                                ec = c3)
        
        ax.add_patch(tp_array)
        
        
        # Axis labels
        ax.set_xlabel('$X$ (in mm)')
        ax.set_ylabel('$Y$ (in mm)')
        
        # Legend
        ax.legend(['Predicted', 'Real', 'Thermopile Array'])
        
        # Title
        ax.set_title('Undistorted (predicted) vs. distorted (real) pixel locations and spatial extent ot thermopile array')
        
        
        return ax
        
    def estimate_mapping(self):
        """
        Estimate mapping for OpenCVs remap() function from GridDistortionData
        """
        
        if self.GridDistortion is not None:
            data = self.GridDistortion.copy()
        else:
            raise Exception('Import Distortion data first.')
            return None
        
        # Get attributes for convenience
        pixpitch = self.pixpitch
        
        
        # Sort by y coordinate first and x coordinate second so reshape can be 
        # applied
        data = data.sort_values(by = ['j','i'],
                                ascending = [True,True])
                
        # Calculate the difference between Real (distorted) and Predicted 
        # (ideal, undistorted)
        # data['dx'] = data['Real X']  - data['Predicted X']
        # data['dy'] = data['Real Y']  - data['Predicted Y']
        data['dx'] = data['Real X']- data['Predicted X']
        data['dy'] = data['Real Y']- data['Predicted Y']
        
        # Create a rectangular grid for the distortion in x and y 
        # direction separately
        
        # Take as coordinates of the grid the predicted coordinates (questionable)
        x_grid = data['Predicted X'].unique()
        y_grid = data['Predicted Y'].unique()
        
        # Get Real X, Y as arrays
        dx = data['dx'].values.reshape((len(y_grid),len(x_grid))).astype(np.float32)
        dy = data['dy'].values.reshape((len(y_grid),len(x_grid))).astype(np.float32)
        
                
        # Create a bilinear grid interpolator object for the x and y distortion 
        # fields
        interp_x = RegularGridInterpolator((y_grid, x_grid),          # note order: (y-axis, x-axis) because D is [y, x]
                                           dx,
                                           method="linear",   # bilinear on a rectilinear grid
                                           bounds_error=False,
                                           fill_value=np.nan
                                        )
        
        interp_y = RegularGridInterpolator((y_grid, x_grid),          # note order: (y-axis, x-axis) because D is [y, x]
                                           dy,
                                           method="linear",   # bilinear on a rectilinear grid
                                           bounds_error=False,
                                           fill_value=np.nan
                                        )
        
        # Create coordinates (mm) for the pixels of the actual thermopile array
        x_tp_pos = np.arange(pixpitch, x_grid.max(), pixpitch)
        y_tp_pos = np.arange(pixpitch, y_grid.max(), pixpitch)
        
        x_tp = np.hstack([-np.flip(x_tp_pos),x_tp_pos])
        y_tp = np.hstack([-np.flip(y_tp_pos),y_tp_pos])
       
        X_tp, Y_tp = np.meshgrid(x_tp,y_tp)
        
        # Pass these query points to the grid interpolators
        dx_tp = interp_x(np.column_stack((Y_tp.flatten(),X_tp.flatten())))
        dy_tp = interp_y(np.column_stack((Y_tp.flatten(),X_tp.flatten())))
        
        # Reshape back
        dx_tp = dx_tp.reshape((len(y_tp),len(x_tp)))
        dy_tp = dy_tp.reshape((len(y_tp),len(x_tp)))
        
        # The whole distortion field is still expressed in mm. Divide by 
        # pixel pitch to convert to pixel
        dx_tp = dx_tp / pixpitch
        dy_tp = dy_tp / pixpitch
        
        # Transform into OpenCVs coordinate sytem to obtain map_x and map_y
        w_new = len(x_tp)
        h_new = len(y_tp)
             
        coords_x, coords_y = np.meshgrid(np.arange(w_new), np.arange(h_new))

        map_x = (coords_x + dx_tp).astype(np.float32)
        map_y = (coords_y + dy_tp).astype(np.float32)
            
        return map_x, map_y
    
    def apply_mapping(self,img_distorted, map_x, map_y, fill = 0.1) -> np.ndarray:
        """
        Apply mapping estimated from Zemax' GridDistortionData to distorted 
        image
        """
        
        if not map_x.shape == map_y.shape:
            raise ValueError('map_x and map_y need to have the same shape!')
        
        # Place distorted image in the center of a new array of the same
        # dimension as the undistorted image
        img_src = np.ones(map_x.shape)*np.nan
        
        h_dist,w_dist = img_distorted.shape
        h_undist,w_unddist = img_src.shape
        
        y_tl = int((h_undist-h_dist)/2)
        y_br = y_tl+h_dist
        
        x_tl = int((w_unddist-w_dist)/2)
        x_br =x_tl+w_dist
        
        img_src[y_tl:y_br,x_tl:x_br] = img_distorted
        
        # Apply remapping
        img_undist = cv2.remap(img_src,
                               map_x,
                               map_y,
                               interpolation=cv2.INTER_CUBIC)
        
        return img_undist
    
    