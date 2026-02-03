# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:15:16 2023

@author: Rehmer
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import os
import socket
import openpyxl 
import matplotlib.pyplot as plt
import re
# import imageio_ffmpeg



# import matplotlib.pyplot as plt
# from scipy.interpolate import LinearNDInterpolator as lNDI

from .tparray import TPArray


        

class QuadriPolygon():
    
    def __init__(self,u):
        
        self.u = ['input1','input2']
        
        # Make sure Tamb0 is always the first input
        if 'Tamb0' in u[0]:
            self.u[0] = u[0]
            self.u[1] = u[1]
        else:
            self.u[0] = u[1]
            self.u[1] = u[0]
            

        
    def derive_polygon(self,df):
        '''
        Derive a quadrilateral polygon from the dataset, which is then used
        to define the limits of the parameter varying part of the model
        '''
    
        # Get the vertices of the quadrilateral polygon by sorting the data
        # and looping through it looking for the most extreme points per
        # measurement series (Ta)
        # v1 = df.loc[df['Ta']==df['Ta'].min()].min()[self.u]
        # v2 = df.loc[df['Ta']==df['Ta'].min()].max()[self.u]
        # v3 = df.loc[df['Ta']==df['Ta'].max()].max()[self.u]
        # v4 = df.loc[df['Ta']==df['Ta'].max()].min()[self.u]


        # convert vertices to DataFrame
        # V = [v1,v2,v3,v4]
        
        V = self._get_vertives(df)
        
        polygon = []
        
        for v in range(4):
            polygon.append(pd.DataFrame(data = [V[v]],
                                  columns=self.u,
                                  index = ['v'+str(v)]))
        
        polygon = pd.concat(polygon)
        
        self.polygon = polygon
        
        return polygon
    
    def _get_vertives(self,df):
        """
        Derives vertices of polygon from the data. 
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        tol = 0.05
        
        T = self.u[0]
        U = self.u[1]
        
        # v0
        Tamb0 = df.loc[df[T]==df[T].min(),T].min()
        Ud0 = df.loc[df[T]<=(1+tol)*Tamb0,U].min()
        
        v0 = pd.Series(data=[Tamb0,Ud0],index=[T,U],name='v0')
        
        # v1
        Tamb0 = df.loc[df[T]==df[T].min(),T].min()
        Ud1 = df.loc[df[T]<=(1+tol)*Tamb0,U].max()
        
        v1 = pd.Series(data=[Tamb0,Ud1],index=[T,U],name='v1')
        
        # v2
        Tamb2 = df.loc[df[T]==df[T].max(),T].max()
        Ud2 = df.loc[df[T]>=(1-tol)*Tamb2,U].max()
        
        v2 = pd.Series(data=[Tamb2,Ud2],index=[T,U],name='v2')

        # v3
        Ud3 = df.loc[df[T]>=(1-tol)*Tamb2,U].min()
        
        v3 = pd.Series(data=[Tamb2,Ud3],index=[T,U],name='v3')        
        
        V = [v0,v1,v2,v3]
        
        return V
    
    # def _in_inner_rect(self,df_pnt):
        
    #     # get polygon
    #     polygon = self.polygon.copy()
        
    #     # Construct corners of a rectangle that lies completely in the polygon
    #     Tamb_low = polygon.loc[['v0','v1'],'Tamb0'].max()
    #     Tamb_up = polygon.loc[['v2','v3'],'Tamb0'].min()
        
    #     Ud_low = polygon.loc[['v0','v3'],'Ud'].max()
    #     Ud_up = polygon.loc[['v1','v2'],'Tamb0'].min()
        
        
    #     # Check if point is inside inner rectangle
    #     if (Tamb_low<df_pnt['Tamb0']<Tamb_up) & (Ud_low<df_pnt['Ud']<Ud_up):
    #         in_poly = True
    #     else:
    #         in_poly = False
            
    #     return in_poly
            
        
        
    def in_polygon(self,df_pnt):
        '''
        This method checks if a point is inside or outside the quadrilateral 
        polygon that defines the area of the input space where data was 
        available for estimating the parameter variation
        
        THIS METHOD WORKS ONLY IF THE POLYGON IS A RECTANGLE
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        
        # dataframe to series
        df_pnt = df_pnt.squeeze()
        
        # Boolean
        in_poly = False

        # get polygon
        polygon = self.polygon.copy()
        
        # Extract scheduling variables from df_pnt
        df_pnt = df_pnt[self.u]
       
        # Edges of the polygon
        E = [('v0','v1'),('v1','v2'),('v2','v3'),('v3','v0')]
        
        # If not, one has to do a long and cumbersome calculation
        # direction of the ray 
        r = df_pnt.copy()
        r[self.u[0]] = 1
        r[self.u[1]] = 1
        
        intersect = []
        
        # Project point on all edges
        for e in E:
                       
            v0 = polygon.loc[e[0]]
            v1 = polygon.loc[e[1]]

            # calculate intersection of the two lines p + k0*r and v0 + k1*(v1-v0)
            # by solving system of linear eqations
            A = np.array([[r[self.u[0]],-(v1-v0)[self.u[0]]],
                         [r[self.u[1]],-(v1-v0)[self.u[1]]]])
            b = (v0-df_pnt).values.flatten()
            
            k = np.linalg.solve(A.astype(float), b.astype(float))
            
            # k0 is parameter of the ray from the point in question
            # k1 is parameter on the edge 
            
            # if k0 is negative the ray does not intersect the line
            # if k1 is negative or larger than one, the ray does not
            # intersect the line
            if k[0]<-10E-6 or k[1]>1 or k[1]<-10E-6:
                intersect.append(False)
            elif k[0] == 0 or k[1]==0:
                    # In the rare case that a point is exactly on an edge,
                    # return True immediately
                    return True
            else:
                intersect.append(True)
            

        # if the number of intersections is odd, the point is inside the polygon
        if sum(intersect)%2 == 1:
            in_poly = True
          
        return in_poly   
            
    def pnt_on_edge(self,df_pnt):
        '''
        Calculates the closest point on the edge of the quadrilateral 
        polygon to a given point   
        
    
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        

        
        # dataframe to series
        pnt = df_pnt.squeeze()
        
        # Extract scheduling variables from df_pnt
        pnt = pnt[self.u]
        
        # Edges of the polygon
        # E = [('v0','v1'),('v1','v2'),('v2','v3'),('v3','v0')]
              
        # list for storing distances in
        dist = []
        
        # List for storing intersection with polygon in
        intersect = []
    
        u0 = self.u[1]#'Ud'
        u1 = self.u[0]#'Tamb0'
        
        # Loop over edges v1,v2 and v3,v0 first
        E = [('v1','v2'),('v3','v0')]
        
        
        # Project point on edges ('v1','v2'),('v3','v0')
        for e in E:
            
            # Get quadrilateral polygon
            df_qpoly = self.polygon.copy()
           
            #Get end points of line segment
            v0 = df_qpoly.loc[e[0]]
            v1 = df_qpoly.loc[e[1]]
        
            # Check if point can be projected on line segment
            
            # Check temperature condition first!
            # if min(v0[u1], v1[u1]) < pnt[u1] < max(v0[u1], v1[u1]):
                
            # Write line segment as line equation Tamb0 = a*Ud
            dv = v1 - v0
            
            # Handle exception of the denominator being zero
            if abs(dv[u0])<1E-6:
                a = 1E-6
            else:
                a = dv[u1]/(dv[u0])
            
            # Write line segment through point as line equation Tamb0 = d
            d = pnt[u1] - v0[u1]
            
            # Point of interesection 
            isect = pd.Series(data=[d/a + v0[u0] ,a*(d/a) + v0[u1]],
                                 index = [u0,u1])
            
            # Save intersection point
            intersect.append(isect)
        
            # Distance between intersection and point
            dist.append(np.linalg.norm(pnt-isect))
        
        # Check if the projection of the point pnt onto the edges of the polygon
        # lies on the edges of the polygon or its "extension" (i.e. a point
        # somewhere on the linear function that describes the edge but beyond
        # the defined polygon)
        if len(dist)>0:
            
            # Find minimal distance
            idx_min = np.array(dist).argmin()
            
            # Get the corresponding intersection point
            isect = intersect[idx_min]
            
            # check if that point is in the polygon
            # if self.in_polygon(isect):
            if self.in_polygon(isect+10E-6*isect) or \
                self.in_polygon(isect-10E-6*isect):
                    
                    
                    # Convert from series to dataframe
                    df_isect = pd.DataFrame(data=[isect.values],
                                            columns=isect.index,
                                            index=df_pnt.index)
                    
                    return df_isect
                
            
        # If no intersection point could be returned, project point on   
        # edge ('v0','v1') and ('v2','v3')
        E = [('v0','v1'),('v2','v3')]
        
        # list for storing distances in
        dist = []
        
        # List for storing intersection with polygon in
        intersect = []
                
        # Project point on edges ('v1','v2'),('v3','v0')
        for e in E:
            # Get quadrilateral polygon
            df_qpoly = self.polygon.copy()
           
            #Get end points of line segment
            v0 = df_qpoly.loc[e[0]]
            v1 = df_qpoly.loc[e[1]]
        
            # Check if point can be projected on line segment
        
            # if min(v0[u0], v1[u0]) < pnt[u0] < max(v0[u0], v1[u0]):
                
            # Write line segment as line equation Ud = a*Tamb0
            dv = v1 - v0
            
            # Write line segment through point as line equation Ud = d
            d = pnt[u0] - v0[u0]
            
            # In case that both vertices have same Ta_coordinate, df[u1] 
            # will become zero. Treat this exception:            
            if dv[u1]==0:
                isect = pd.Series(data=[0+v0[u1] ,d + v0[u0]],
                                     index = [u1,u0])
            else: 
                a = dv[u0]/dv[u1]
                isect = pd.Series(data=[d/a + v0[u1] ,a*(d/a) + v0[u0]],
                                     index = [u1,u0])
            
            # Save intersection point
            intersect.append(isect)
        
            # Distance between intersection and point
            dist.append(np.linalg.norm(pnt-isect))
        
        if len(dist)>0:
            idx_min = np.array(dist).argmin()
            
            # Get the corresponding intersection point
            isect = intersect[idx_min]
            
            # Check if intersection is in polygon 
            if self.in_polygon(isect+10E-6*isect) or \
                self.in_polygon(isect-10E-6*isect):
                    # Convert from series to dataframe
                    df_isect = pd.DataFrame(data=[isect.values],
                                            columns=isect.index,
                                            index=df_pnt.index)
                    
                    return df_isect
                 
        # In the case that projection on the edge was not possible, 
        # find the closest vertex      
        
        # list for storing distances in
        dist = []
        
        # List for storing intersection with polygon in
        intersect = []
        
        df_qpoly['d_Tamb0'] = abs(df_qpoly[u1] - df_pnt[u1].item())
        
        # Sort by difference and keep the first two
        df_qpoly = df_qpoly.sort_values(by='d_Tamb0')
        
        df_qpoly = df_qpoly.iloc[0:2]
        
        # Among these two vertices find the closest            
        vert = [df_qpoly.loc[v] for v in df_qpoly.index]
        
        # Find vertex closest 
        d =  [np.linalg.norm(pnt[self.u]-v[self.u]) for v in vert]
                   
        d = np.array(d)
        
        # Distance between closest vertice and point
        dist.append(min(d))
        
        # Save intersection point
        isect = (vert[d.argmin()]).copy()
        intersect.append(isect[self.u])
        
        df_isect = pd.DataFrame(data=[isect.values],
                                columns=isect.index,
                                index=df_pnt.index)
         

         
        return df_isect
            
    
             
    def pnt_on_edge_old(self,df_pnt):
        '''
        Calculates the closest point on the edge of the quadrilateral 
        polygon to a given point   
        
    
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        

        
        # dataframe to series
        pnt = df_pnt.squeeze()
        
        # Extract scheduling variables from df_pnt
        pnt = pnt[self.u]
        
        # Edges of the polygon
        E = [('v0','v1'),('v1','v2'),('v2','v3'),('v3','v0')]
              
        # list for storing distances in
        dist = []
        
        # List for storing intersection with polygon in
        intersect = []
        

     
        
        # Project point on all edges
        for e in E:
            
            # Get quadrilateral polygon
            df_qpoly = self.polygon.copy()
           
            #Get end points of line segment
            v0 = df_qpoly.loc[e[0]]
            v1 = df_qpoly.loc[e[1]]
            

            # Check if point can be projected on line segment
            u0 = 'Ud_pv'
            u1 = 'Tamb0_pv'
            
            # Check temperature condition first!
            if min(v0[u1], v1[u1]) < pnt[u1] < max(v0[u1], v1[u1]):
                
                # Write line segment as line equation Tamb0 = a*Ud
                dv = v1 - v0
                a = dv[u1]/dv[u0]
                
                # Write line segment through point as line equation Tamb0 = d
                d = pnt[u1] - v0[u1]
                
                # Point of interesection 
                isect = pd.Series(data=[d/a + v0[u0] ,a*(d/a) + v0[u1]],
                                     index = [u0,u1])
                
                # Check if intersection is in polygon 
                if self.in_polygon(isect+0.001*isect) or \
                    self.in_polygon(isect-0.001*isect):
                    
                        dd = np.linalg.norm(pnt-isect)    
                        
                        # check if any vertice is closer in Ta-direction than
                        # intersection point
                        # if v0[u1] - pnt[u1] < dd:
                        #     isect = v0
                        # elif v1[u1] - pnt[u1] < dd:
                        #     isect = v1
                        
                        # Save intersection point
                        intersect.append(isect)
                    
                        # Distance between intersection and point
                        dist.append(np.linalg.norm(pnt-isect))
            
            elif min(v0[u0], v1[u0]) < pnt[u0] < max(v0[u0], v1[u0]):
                
                # Write line segment as line equation Ud = a*Tamb0
                dv = v1 - v0
                # Write line segment through point as line equation Ud = d
                d = pnt[u0] - v0[u0]
                
                # In case that both vertices have same Ta_coordinate, df[u1] 
                # will become zero. Treat this exception:
                if dv[u1]==0:
                    isect = pd.Series(data=[0+v0[u1] ,d + v0[u0]],
                                         index = [u1,u0])
                
                else: 
                    a = dv[u0]/dv[u1]
                    isect = pd.Series(data=[d/a + v0[u1] ,a*(d/a) + v0[u0]],
                                         index = [u1,u0])
                
                
                # Check if intersection is in polygon 
                if self.in_polygon(isect+0.001*isect) or \
                    self.in_polygon(isect-0.001*isect):
                        
                        dd = np.linalg.norm(pnt-isect)    
                        
                        # check if any vertice is closer in Ta-direction than
                        # intersection point
                        if abs(v0[u1] - pnt[u1]) < dd:
                            isect = v0
                        elif abs(v1[u1] - pnt[u1]) < dd:
                            isect = v1
                        
                        # Save intersection point
                        intersect.append(isect)

                        # Distance between intersection and point
                        dist.append(np.linalg.norm(pnt-isect))
                
                

    
                
                        
        # If intersection with line is not possible, find closest
        # vertice
        if len(intersect)==0:
            
            df_qpoly['d_Tamb0'] = abs(df_qpoly[u1] - df_pnt[u1].item())
            
            # Sort by difference and keep the first two
            df_qpoly = df_qpoly.sort_values(by='d_Tamb0')
            
            df_qpoly = df_qpoly.iloc[0:2]
            
            # Among these two vertices find the closest            
            vert = [df_qpoly.loc[v] for v in df_qpoly.index]
            
            # Find vertice closest 
            d =  [np.linalg.norm(pnt[self.u]-v[self.u]) for v in vert]
                       
            d = np.array(d)
            
            # Distance between closest vertice and point
            dist.append(min(d))
            
            # Save intersection point
            isect = (vert[d.argmin()]).copy()
            intersect.append(isect[self.u])
                
            
            
            
        # In the end find and return the intersection point with the
        # smallest distance
        # Find minimal distance
        idx_min = np.array(dist).argmin()
        
        # Return the corresponding intersection point
        df_isect = intersect[idx_min]
        
        # Convert from series to dataframe
        df_isect = pd.DataFrame(data=[df_isect.values],
                                columns=df_isect.index,
                                index=df_pnt.index)

        
        return df_isect           
            
    def orth_proj_on_edge(self,df_pnt):
        '''
        Calculates the orthogonal projection of a point on the edge of the 
        quadrilateral polygon to a given point   
        
    
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        # Get quadrilateral polygon
        df_qpoly = self.polygon
        
        # dataframe to series
        pnt = df_pnt.squeeze()
        
        # Extract scheduling variables from df_pnt
        pnt = pnt[self.u]
        
        # Edges of the polygon
        E = [('v0','v1'),('v1','v2'),('v2','v3'),('v3','v0')]
        
        #
        orth_dist = []
        intersect = []
        
        # Project point on all edges
        for e in E:
            v0 = df_qpoly.loc[e[0]]
            v1 = df_qpoly.loc[e[1]]
            
            #calculate the orthgonal distance to the edge
            a = v1 - v0
            b = pnt - v0
            
            norm_a = np.linalg.norm(a)
            
            d = abs(np.cross(a,b)/norm_a)
            
            # Calculate intersection
            isect = np.dot(a,b)/norm_a * (a/norm_a) + v0
            
            # If intersection is not on the edge (in polygon)
            if not self.in_polygon(isect):
                # calculate distances to vertices
                d1 = np.linalg.norm(pnt-v0)
                d2 = np.linalg.norm(pnt-v1)
                
                d = min([d1,d2])
                isect = list([v0,v1])[[d1,d2].index(d)]
                
            orth_dist.append(d)
            intersect.append(isect)
        
        # Find minimal distance
        idx_min = np.array(orth_dist).argmin()
        
        # Return the corresponding intersection point
        df_isect = intersect[idx_min]
        
        # Convert from series to dataframe
        df_isect = pd.DataFrame(data=[df_isect.values],
                                columns=df_isect.index,
                                index=df_pnt.index)    
        return df_isect

def pd_interpolate_MI (df_input, df_toInterpolate,col):
    
    #create the function of interpolation
    func_interp = lNDI(points=df_input.index.to_frame().values, 
                       values=df_input[col].values)
    #calculate the value for the unknown index
    df_toInterpolate[col] = func_interp(df_toInterpolate.index.to_frame().values)
    #return the dataframe with the new values
    return pd.concat([df_input, df_toInterpolate]).sort_index()



    

    

        
        
        
        
        
        