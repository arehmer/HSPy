# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:11:36 2024

@author: rehmer
"""
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from warnings import warn



class LuT:
    
    def __init__(self,**kwargs):
        
        self._LuT = None                    # Look-up-Table as pd.DataFrame
        self._LuTGridInterpolator = None    # SciPy Regular Grid Interpolator, generated automatically  
        
        pass
    
    @property
    def LuT(self):
        return self._LuT
    @LuT.setter
    def LuT(self,lut:pd.DataFrame):
        if not isinstance(lut,pd.DataFrame):
            raise TypeError(f'lut is not type {pd.DataFrame} but {type(lut)}.')
        else:
            self._LuT = lut

    @property
    def LuTGridInterpolator(self):
        return self._LuTGridInterpolator
    @LuTGridInterpolator.setter
    def LuTGridInterpolator(self,interpol:RegularGridInterpolator):
        if not isinstance(interpol,RegularGridInterpolator):
            raise TypeError(f'interpol is not type {type(RegularGridInterpolator)} but {type(interpol)}.')
        else:
            self._LuTGridInterpolator = interpol
    
    def from_df(self,df):
        
        # Set class property        
        self.LuT = df
        
        # Create LuTGridInterpolator for fast bilinear interpolation
        self._create_LuTGridInterpolator()
        
    def from_csv(self,csv_path,offset):
        
        self.offset = offset
        
        # Import data
        LuT = pd.read_csv(csv_path, sep=',',header=0,index_col = 0)
        
        # Convert column header to int
        LuT.columns = np.array([int(c) for c in LuT.columns])
        
        # Subtract offset
        LuT.index = LuT.index - offset
        
        # Set class property   
        self.LuT = LuT
        
        # Create LuTGridInterpolator for fast bilinear interpolation
        self._create_LuTGridInterpolator()

    def from_HTPAxls(self,sheet_name,**kwargs):
        print('Loading LuT ' + sheet_name + ' ...')
        xlsx_standard = Path('T:/Projekte/HTPA8x8_16x16_32x31/Datasheet/LookUpTablesHTPA.xlsm')
        
        xlsx_path = kwargs.pop('xlsx_path',xlsx_standard)
        self.from_xlsx(xlsx_path,sheet_name,**kwargs)
        print(sheet_name + ' successfully loaded.')
        return None
        
    def from_xlsx(self,xlsx_path,sheet_name,**kwargs):
        
        index_col = kwargs.pop('index_col',0)
        usecols = kwargs.pop('usecols','A,D:P')
        skiprows = kwargs.pop('skiprows',15)
        header = kwargs.pop('header',0)
        # dtype = kwargs.pop('dtype','object')
        
        df = pd.read_excel(xlsx_path,
                           sheet_name = sheet_name,
                           skiprows = skiprows,
                           index_col = index_col,
                           usecols = usecols,
                           header = header)
        
        # Delete all commata
        df = df.replace(',','',regex=True)
        
        # Cast all columns to int
        df = df.astype(np.int32)
        
        # Rename index
        df.index.name = 'Ud'
        
        # Set class property   
        self.LuT = df
        
        # Create LuTGridInterpolator for fast bilinear interpolation
        self._create_LuTGridInterpolator()
        
        return None
    
    
    def _create_LuTGridInterpolator(self):
        """
        Creates a scipy regular grid interpolator for fast bilinear interpola-
        tion over the LuT
        
        Source https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html#scipy.interpolate.RegularGridInterpolator

        Returns
        -------
        None.

        """
        
        # Check if property LuT with pd.DataFrame as LuT has been set
        if self.LuT is None:
            raise ValueError('No imported LuT found, self.LuT is None.')
    
        
        # Get all Ta values and Ud values from LuT
        Ta_elements = self.LuT.columns
        Ud_elements = self.LuT.index.values
        
        # Create a 2d grid from Ta_elements and Ud_elements
        # Ta_grid, Ud_grid = np.meshgrid(Ta_elements,Ud_elements)
        
        # Get all To values from the LuT
        To_elements = self.LuT.values
        
        # Make sure the created grids and To_elements have the same shape
        # if not Ta_grid.shape == Ud_grid.shape == To_elements.shape:
        #     raise ValueError('Shapes of LuT and created Ta-Ud-grids are not consistent.')
        
        # Create RegularGridInterpolator
        interp = RegularGridInterpolator((Ud_elements,Ta_elements),
                                         To_elements)
        
        # Assign to class property
        self.LuTGridInterpolator = interp
    
    
    def to_xlsx(self,xlsx_path):
        
        # Load LuT
        LuT = self.LuT.copy()
        
        # Initialize writer object
        writer = self._init_xlswriter(xlsx_path)
        
        # Start a list where columns are in the right order
        columns_ordered = list(LuT.columns)
        
        # Reindex so Ud becomes column
        LuT = LuT.reset_index(drop=False)        
        
        # Add a column vector for the offset-free voltage signal
        LuT['Ud_norm'] = LuT['Ud'] - LuT['Ud'].min() 
        
        # Opening and closing brackets
        LuT['br_o'] = ['{']*len(LuT)
        LuT['br_c'] = ['},']*len(LuT)
        
        columns_ordered = ['Ud','Ud_norm','br_o'] + columns_ordered + ['br_c']
        
        LuT = LuT[columns_ordered]
        
        LuT.to_excel(writer,
                     sheet_name = xlsx_path.stem,
                     startrow=15,
                     index=False)
        
        # Then write normalized Ud to row 9
        df_V = pd.DataFrame(data=[],
                            columns=np.arange(0,len(LuT),1,dtype=int),
                            index=['V'])
        
        columns_ordered = list(df_V.columns)
                
        df_V.loc['V'] = LuT['Ud_norm'].values.astype(int)
        
        df_V = df_V.astype('str')
        df_V[df_V.columns[:-1]] = df_V[df_V.columns[:-1]]+','
        
        df_V['br_o'] = '{'
        df_V['br_c'] = '};'

        columns_ordered =  ['br_o'] + columns_ordered + ['br_c']
        
        df_V = df_V[columns_ordered]
        
        df_V.to_excel(writer,
                      sheet_name = xlsx_path.stem,
                      startrow=12,
                      index=True,
                      header=False)
        
        # and Ta to row 10
        df_Ta = pd.DataFrame(data=[],
                            columns=list(self.LuT.columns),
                            index=['Ta'])
        
        columns_ordered = list(df_Ta.columns)
                
        df_Ta.loc['Ta'] = list(self.LuT.columns)
        

        
        df_Ta = df_Ta.astype('str')
        df_Ta[df_Ta.columns[:-1]] = df_Ta[df_Ta.columns[:-1]]+','

        
        df_Ta['br_o'] = '{'
        df_Ta['br_c'] = '};'

        columns_ordered =  ['br_o'] + columns_ordered + ['br_c']
        
        df_Ta = df_Ta[columns_ordered]
        
        df_Ta.to_excel(writer,
                       sheet_name = xlsx_path.stem,
                       startrow=13,
                       index=True,
                       header=False)
        
        writer.close()
            

        
    def inverse_eval_LuT(self,data:pd.DataFrame,
                         Ta:str='Tamb0',
                         To:str='To_meas')->pd.DataFrame:
        """
        Converts measurements given in Kelvin back to Voltage in Digits.

        Parameters
        ----------
        data : pd.DataFrame
            DESCRIPTION.
        Ta_col : str, optional
            Column label of column containing ambient temperature in Kelvin. 
            The default is 'Tamb0'.
        To_col : str, optional
            Column label of column containing measured object temperature in 
            Kelvin. The default is 'To_meas'.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
        # Check if index is unique, otherwise loop will exract more than one
        # measurement per loop and algorithm brakes down
        if not data.index.is_unique:
            print('Data index is not unique. reset_index() is applied!')
            data = data.reset_index()
        
        # print('Temperatures are assumed to be provided in Kelvin ' +\
        #       'and are converted to dK.')
        
        
        data[[Ta,To]] = (data[[Ta,To]]*10).astype(int)
        
        Ud_LuT = []
        
        for meas in data.index:

            LuT = self.LuT
        
            Ta_meas = data.loc[meas,Ta]
            To_meas = data.loc[meas,To]
            
            # find the "last" column in old LuT that is smaller than the 
            # measured Ta 
            Ta_iloc = np.where(LuT.columns < Ta_meas)[0][-1]
            Ta_cols = LuT.columns[Ta_iloc:Ta_iloc+2]
                      
            # create a column for the actually measured Tamb0 via interpolation
            Tamb0  = int(np.round(Ta_meas))
            f = (Tamb0-Ta_cols[0]) / (Ta_cols[1]-Ta_cols[0])
            
            Tamb0_col = LuT[Ta_cols[0]] + \
                (f*(LuT[Ta_cols[1]]-LuT[Ta_cols[0]])).astype(np.int32)
            
            
            # Find index of last To in that column that is smaller than the 
            # measured To and the next one (which is the first that is larger)
            Ud_iloc = np.where(Tamb0_col.values<To_meas)[0][-1]
            Ud_1 = Tamb0_col.index[Ud_iloc]
            Ud_2 = Tamb0_col.index[Ud_iloc+1]
            
            Ta_1 = Tamb0_col.loc[Ud_1]
            Ta_2 = Tamb0_col.loc[Ud_2]
            
            dT = Ta_2 - Ta_1
            dU = Ud_2 - Ud_1
            
            f =  dU/dT
            
            Ud_LuT.append(Ud_1 +  (To_meas - Ta_1) * f)
            
               

        data['Ud_LuT'] = Ud_LuT
        
        # print('Temperatures are converted back to Kelvin.')
        data[[Ta,To]] = (data[[Ta,To]]/10)
        
        return None
        
    def eval_LuT(self,data:pd.DataFrame,Ta_col='Tamb0',Ud_col='Ud'):
        """
        

        Parameters
        ----------
        data : pd.DataFrame. Contains ambient temperature in Ta_col in dK
            and pixel voltage in digits in Ud_col
            DESCRIPTION.
        Ta_col : TYPE, optional
            DESCRIPTION. The default is 'Tamb0'.
        Ud_col : TYPE, optional
            DESCRIPTION. The default is 'Ud'.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
        
        DeprecationWarning('This function will be superseded by calc_To() ' +\
                           'in the future.')
        
        LuT = self.LuT
        
        # Convert measurements in Kelvin to dK
        data[[Ta_col]] = (data[[Ta_col]]*10).astype(int)
        
        for meas in data.index:
                     
            Ta_meas = data.loc[meas,Ta_col]
            Ud = data.loc[meas,Ud_col]
        
            # Find columns and indices for bilinear interpolation
            col_idx = LuT.columns < Ta_meas
            
            # Check if Ta_meas is outside of the range of the LuT
            if not (all(col_idx==True) or all(col_idx==False)):
                LuT_col = LuT.columns[col_idx][-1]
                Ta_col_n = LuT.columns[LuT.columns.get_loc(LuT_col)+1]
            else:
                # If so, write error value to data and skip
                data.loc[meas,'To_LuT'] = -99
                continue
            
            # Check if Ud is outside of the range of the LuT
            row_idx = LuT.index < Ud
            if all(row_idx==True) or all(row_idx==False):
                # If so, write error value to data and skip
                data.loc[meas,'To_LuT'] = -99
                continue
            
            
            Ud_row = LuT.index[row_idx][-1]
            Ud_row_n = LuT.index[LuT.index.get_loc(Ud_row)+1]
            
            
            rect_pnts = LuT.loc[Ud_row:Ud_row_n,[LuT_col,Ta_col_n]]
            
            data.loc[meas,'To_LuT'] = \
                self._get_To(rect_pnts,Ta_meas,Ud)
                
        # Convert dK back to K
        data[[Ta_col,'To_LuT']] = (data[[Ta_col,'To_LuT']]/10)
                        
        return data
        
    def calc_To(self,
                qpoints: tuple | np.ndarray | pd.Series | pd.DataFrame ) -> List[float]:
        """
        Determines To for a given pair of (Ud,Tamb0). Ud must be given in Digits,
        Ta must be given in dK.
        

        Parameters
        ----------
        qpoints : tuple | pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        List[float]
            DESCRIPTION.

        """
        
        # ---------- Query Point is given as a tuple --------------------------
        if isinstance(qpoints,tuple) | isinstance(qpoints,np.ndarray):
            return self._calc_To(qpoints)
        
        # ---------- Query Points are given as a np.ndarray -------------------
        if isinstance(qpoints,np.ndarray):
            return self._calc_To(qpoints)
        
        # ------------ Query Points are given as pd.Series --------------------
        if isinstance(qpoints,pd.Series):
            
            # Check if index names associated with Ud and Tamb do exist
            if not 'Ud' in qpoints.index:
                raise ValueError(f"Index 'Ud' not in {qpoints}")
            if not 'Tamb0' in qpoints.index:
                raise ValueError(f"Index 'Tamb0' not in {qpoints}")
            
            return self._calc_To( (qpoints['Ud'],qpoints['Tamb']) )
        
        # --------- Query Points are given as pd.DataFrame --------------------
        if isinstance(qpoints,pd.DataFrame):
            
            # Check if index names associated with Ud and Tamb do exist
            if not 'Ud' in qpoints.columns:
                raise ValueError(f"Index 'Ud' not in {qpoints.columns}")
            if not 'Tamb0' in qpoints.columns:
                raise ValueError(f"Index 'Tamb0' not in {qpoints.columns}")
            
            return self._calc_To( qpoints[['Ud','Tamb0']].values)
                       
            
    def _calc_To(self,
                 qpoints : tuple | np.ndarray) -> tuple | np.ndarray:
        """
        Low-level helper of user interface method calc_To
        """
        
        # ---------- Query Point is given as a tuple --------------------------
        if isinstance(qpoints,tuple):
            Ud = qpoints[0]
            Ta = qpoints[1]
            
            try:
                To = self.LuTGridInterpolator((Ud,Ta))
                
                
            except ValueError:
                
                if not self.LuT.index.values.min() <= Ud <= self.LuT.index.values.max():
                    warn('Queried Ud value is out of bounds. To=-1 returned.\n')
                    To = -1

                if not self.LuT.columns[0] <= Ta <= self.LuT.columns[-1]:
                    warn('Queried Tamb0 value is out of bounds, To=-2 returned.\n')
                    To = -2
                    
            return To
        
        
        # ---------- Query Points are given as a np.ndarray -------------------
        if isinstance(qpoints,np.ndarray):
            
            n_cols = qpoints.shape[1]
            
            if n_cols !=2:
                raise ValueError('shape mismatch: qpoints must have shape ' +\
                                 '[(Ud_0,Tamb0_0),..., (Ud_N,Tamb0_N)] with ' +\
                                     'N: number of query points.')
                    
            try:
                # Try to evaluate whole array in one go
                To = self.LuTGridInterpolator(qpoints)
            
            except ValueError:
                # If only one query point is out of bounds, a ValueError is 
                 # returned. In that case, try to evalute point by point
                To = []
                for qpoint in qpoints:
                    To.append(self._calc_To(tuple(qpoint)))
            
            return To
        

    def _get_To(self,points,Ta_meas,Ud_meas):
        
        x0 = points.columns[0]
        x1 = points.columns[1]
        
        y0 = points.index[0]
        y1 = points.index[1]
    
        
        pt1 = (x0,y0,points.loc[y0,x0])
        pt2 = (x0,y1,points.loc[y1,x0])
        pt3 = (x1,y0,points.loc[y0,x1])
        pt4 = (x1,y1,points.loc[y1,x1])
            
        rect_pnts = np.array([pt1,pt2,pt3,pt4])
        
        return self._bilinear_interpolation(Ta_meas,Ud_meas,rect_pnts)
        
    def _bilinear_interpolation(self,x, y, points):
        '''Interpolate (x,y) from values associated with four points.
    
        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.
    
            >>> bilinear_interpolation(12, 5.5,
            ...                        [(10, 4, 100),
            ...                         (20, 4, 200),
            ...                         (10, 6, 150),
            ...                         (20, 6, 300)])
            165.0
    
        '''
        # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    
        # points = sorted(points)               # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
    
        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')
    
        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
               ) / ((x2 - x1) * (y2 - y1) + 0.0)
        
    def plot_LuT(self):
        
        LuT = self.LuT
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        X = np.array(LuT.columns).reshape((1,-1))
        X = np.repeat(X,len(LuT),axis=0)
        
        Y = np.array(LuT.index).reshape((-1,1))
        Y = np.repeat(Y,len(LuT.columns),axis=1)
        
        Z = LuT.values
        
                
        ax.plot_surface(X,Y,Z,cmap=matplotlib.cm.coolwarm,antialiased=False)
        ax.set_xlabel('Tamb0')
        ax.set_ylabel('Ud')
        ax.set_zlabel('To_pred')
        
        pass
            
    def _init_xlswriter(self,path):
        
        # Load or create workbook
        writer = pd.ExcelWriter(path, engine='openpyxl')
        writer.book.create_sheet(title=path.stem)
        
        return writer