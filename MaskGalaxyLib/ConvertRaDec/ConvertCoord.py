import math as m
import numpy as np
import pandas as pd
from scipy.spatial import distance
from astropy.io import fits as pyfits,ascii
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse
import matplotlib.pyplot  as plt
from matplotlib.pyplot import figure
from matplotlib.patches import Circle
from urllib.request import urlopen
from bs4 import BeautifulSoup
from pandas import read_html, DataFrame
import requests
import skimage.io
from astropy.table import Table, Column, join, vstack, hstack

class ConvertCoord():

    data =None;ra_im_center=None;dec_im_center=None
    im_scale=None;im_pixel=None;im_size=None
    x_im_center=None;y_im_center=None;sr=None
    sr_use=None;config=None

    class_array = ['BG','S','E']

    def __init__(self, _pixel, _df, _RA, _DEC, _scale, _config):
        self.data = _df #pd.DataFrame(data=_df,columns=['x', 'y', 'brightness', 'class', 'Prob'])
        #self.data.columns=['x','y', 'brightness', 'class', 'Prob']
        self.ra_im_center=_RA          #degree
        self.dec_im_center=_DEC        #degree
        self.im_scale=float(_scale)           #arcsec/pixel
        self.im_pixel=_pixel           #pixels
        self.im_size=float(self.im_scale)*float(self.im_pixel) #arcsec
        self.x_im_center=float(self.im_pixel)/2    #pixels
        self.y_im_center=float(self.im_pixel)/2    #pixels
        self.sr=float(self.im_size)/2/60  #arcmin
        self.sr_use=round(float(self.sr)+0.9,1)
        self.config = _config

    
    def anti_clockwise(self,x,y):
        alpha = np.degrees(np.arctan2(y,x))
        return (alpha + 360) % 360

    def getVOTable(self):
        fltr = self.config['url_skyserver_votable']+'ra='+str(self.ra_im_center)+'&dec='+str(self.dec_im_center)+'&sr='+str(self.sr_use)
        res = requests.get(fltr)
        soup = BeautifulSoup(res.text,'html.parser')
        table = soup.find_all('table')
        df=read_html(str(table), flavor='html5lib')
        return df

    def getConvertCoord(self,PathFile="",ReturnOpt="file"):
        angle_xy=[1.0]*len(self.data);dist_arcsec=[1.0]*len(self.data);dist_pix=[1.0]*len(self.data)
        for i in range(len(self.data)):
            dist_pix[i]=distance.euclidean([self.x_im_center,self.y_im_center,0],[self.data['x'][i],self.data['y'][i],0])
            dist_arcsec[i]= dist_pix[i]*self.im_scale
            angle_xy[i]= self.anti_clockwise(self.data['x'][i]-self.x_im_center,self.data['y'][i]-self.y_im_center)

        df = self.getVOTable()

        RA_sdss       = df[0][1]  #pd['RA']
        DEC_sdss      = df[0][2]  #pd['DEC']
        type_sdss     = df[0][3]  #pd['TYPE']
        u_psf_sdss    = df[0][14] #pd['PSFMAG_U']
        u_psf_sdss_err= df[0][15] #pd['PSFMAGERR_U']
        g_psf_sdss    = df[0][16] #pd['PSFMAG_G']
        g_psf_sdss_err= df[0][17] #pd['PSFMAGERR_G']
        r_psf_sdss    = df[0][18] #pd['PSFMAG_R']
        r_psf_sdss_err= df[0][19] #pd['PSFMAGERR_R']
        i_psf_sdss    = df[0][20] #pd['PSFMAG_I']
        i_psf_sdss_err= df[0][21] #pd['PSFMAGERR_I']
        z_psf_sdss    = df[0][22] #pd['PSFMAG_Z']
        z_psf_sdss_err= df[0][23] #pd['PSFMAGERR_Z']

        dist_arcsec_sdss=[1.0]*len(RA_sdss)
        object_angle=[1.0]*len(RA_sdss)
        diff_dist=[1.0]*len(RA_sdss)
        diff_arcsec_sdss=[1.0]*len(self.data)
        object_ra=[1.0]*len(self.data)
        object_dec=[1.0]*len(self.data)
        object_x=[1.0]*len(self.data)
        object_y=[1.0]*len(self.data)
        object_class=[1.0]*len(self.data)
        object_prob=[1.0]*len(self.data)
        object_angle_xy=[1.0]*len(self.data)
        object_angle_sdss=[1.0]*len(self.data)
        object_type_sdss=[1.0]*len(self.data)
        object_u_psf=[1.0]*len(self.data)
        object_u_psf_err=[1.0]*len(self.data)
        object_g_psf=[1.0]*len(self.data)
        object_g_psf_err=[1.0]*len(self.data)
        object_r_psf=[1.0]*len(self.data)
        object_r_psf_err=[1.0]*len(self.data)
        object_i_psf=[1.0]*len(self.data)
        object_i_psf_err=[1.0]*len(self.data)
        object_z_psf=[1.0]*len(self.data)
        object_z_psf_err=[1.0]*len(self.data)
        angle_interval=3. # +/- 
        cut_diff_dist=2.  #arcsec
 
        for i in range(len(self.data)):
            id_obj=[]
            
            for j in range(len(RA_sdss)):        
                c1 = SkyCoord(self.ra_im_center*u.deg, self.dec_im_center*u.deg, frame='icrs')
                c2 = SkyCoord(RA_sdss[j]*u.deg, DEC_sdss[j]*u.deg, frame='icrs')
                
                dist_temp=c1.separation(c2)
                dist_arcsec_sdss[j]=dist_temp.arcsecond
                diff_dist[j]=(dist_arcsec[i] - dist_arcsec_sdss[j])

                ra_new=self.ra_im_center-RA_sdss[j]
                dec_new=self.dec_im_center-DEC_sdss[j]
            
                object_angle[j]=self.anti_clockwise(ra_new,dec_new)
                if( object_angle[j]-angle_interval  <= angle_xy[i] <= object_angle[j]+angle_interval and np.absolute(diff_dist[j])<cut_diff_dist):
                    id_obj.append(j)
                else: 
                    object_x[i]=self.data['x'][i]
                    object_y[i]=self.data['y'][i]
                    object_prob[i]=round(self.data['Prob'][i],5)
                    object_class[i]=self.class_array[self.data['class'][i]]
                    object_ra[i]='-'
                    object_dec[i]='-'
                    object_type_sdss[i]='-'
                    object_u_psf[i]='-'
                    object_u_psf_err[i]='-'
                    object_g_psf[i]='-'
                    object_g_psf_err[i]='-'
                    object_r_psf[i]='-'
                    object_r_psf_err[i]='-'
                    object_i_psf[i]='-'
                    object_i_psf_err[i]='-'
                    object_z_psf[i]='-'
                    object_z_psf_err[i]='-'
                        
            diff_dist_new=[]
            for j in id_obj: diff_dist_new.append(diff_dist[j])
            for j in id_obj:
                if( np.absolute(diff_dist[j]) == min(np.absolute(diff_dist_new))):
                    diff_arcsec_sdss[i]=diff_dist[j]
                    object_ra[i]=RA_sdss[j]
                    object_dec[i]=DEC_sdss[j]
                    object_x[i]=self.data['x'][i]
                    object_y[i]=self.data['y'][i]
                    object_class[i]=self.class_array[self.data['class'][i]]
                    object_prob[i]=round(self.data['Prob'][i],5)
                    object_angle_xy[i]=angle_xy[i]
                    object_angle_sdss[i]=object_angle[j]
                    object_type_sdss[i]=type_sdss[j]
                    object_u_psf[i]=round(u_psf_sdss[j],5)
                    object_u_psf_err[i]=u_psf_sdss_err[j]
                    object_g_psf[i]=round(g_psf_sdss[j],5)
                    object_g_psf_err[i]=g_psf_sdss_err[j]
                    object_r_psf[i]=round(r_psf_sdss[j],5)
                    object_r_psf_err[i]=r_psf_sdss_err[j]
                    object_i_psf[i]=round(i_psf_sdss[j],5)
                    object_i_psf_err[i]=i_psf_sdss_err[j]
                    object_z_psf[i]=round(z_psf_sdss[j],5)
                    object_z_psf_err[i]=z_psf_sdss_err[j]  
        if(ReturnOpt=="file"):         
            data_dt = Table({'x': object_x,'y': object_y,'ra':object_ra,'dec':object_dec,'class':object_class,'prob':object_prob,'type':object_type_sdss,'u': object_u_psf ,'g': object_g_psf,'r': object_r_psf,'i': object_i_psf,'z': object_z_psf},names=['x','y','ra','dec','class','prob','type','u','g','r','i','z'])
            ascii.write(data_dt, PathFile, format='fixed_width', delimiter=None)
        elif(ReturnOpt=="df"):
            return pd.DataFrame({'x':object_x,'y': object_y,'ra':object_ra,'dec':object_dec,'class':object_class,'prob':object_prob,'type':object_type_sdss,'u': object_u_psf ,'g': object_g_psf,'r': object_r_psf,'i': object_i_psf,'z': object_z_psf})

  