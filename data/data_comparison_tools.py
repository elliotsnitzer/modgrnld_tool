#!/usr/bin/env python

from IPython.display import Markdown, display
import os, sys, shutil, json, stat
from hublib.ui import Download
import ipywidgets as widgets

import time

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

import math

import glob
from datetime import date
import calendar

from netCDF4 import Dataset
import numpy as np

import pyproj
if pyproj.__version__[0] == '1':
    from pyproj import Proj, transform
elif pyproj.__version__[0] == '2' or pyproj.__version__[0] == '3':
    from pyproj import Transformer

sys.path.append('data')
import nsidc_download_tools
import data_comparison_tools
import raster
import bilinear_interpolate

import requests

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import mean_squared_error

from math import sqrt

import xskillscore as xs

from scipy.stats import pearsonr

import xarray as xr

from multiprocessing import Process, Queue, Lock

def set_variables(cscope, cvar, cy, cm, cv, fn, dD, MERRAut, MERRAft, dp, mc):
    global comparison_scope, comparison_variable, comparison_year, comparison_month, comparison_version, file_number, dataDir, MERRA_url_template, MERRA_filename_template, debug_plots, mask_colors
    comparison_scope = cscope
    comparison_variable = cvar
    comparison_year = cy
    comparison_month = cm
    comparison_version = cv
    file_number = fn
    dataDir = dD
    MERRA_url_template = MERRAut
    MERRA_filename_template = MERRAft
    debug_plots = dp
    mask_colors = mc

    
# Download MODGRNLD data from NSIDC 
def download_nsidc_data():
    global MODGRNLD_url_list, ncfilename
    
    # Find all files that match
    filename_filter = 'MODGRNLD.{:4d}{:02d}.monthly.{}.nc'.format(comparison_year, comparison_month, comparison_version)
    MODGRNLD_url_list = nsidc_download_tools.cmr_search('MODGRNLD', filename_filter=filename_filter)

    job_output_file = open('job_info.txt','w')
    # Look for the files in the data directory
    MODGRNLD_url_list_download = list()
    for url in MODGRNLD_url_list:
        filename = url.split('/')[-1]
        job_output_file.write(filename+'\n')
        if not os.path.exists(dataDir + '/MODGRNLD/' + filename):
            MODGRNLD_url_list_download.append(url)
        else:
            print('{:60s} already downloaded'.format(filename))

    if len(MODGRNLD_url_list_download) > 0:
        # Download the files to the current directory
        #nsidc_download_tools.cmr_download(MODGRNLD_url_list_download)
        
        ##Prallel Downloads##
        ###########################################
        url_count = len(MODGRNLD_url_list_download)
        print('Downloading {0} files...'.format(url_count))
        
        downloads = list()
        for index, url in enumerate(MODGRNLD_url_list_download, start = 1):
            # Download the files to the current directory
            filename = url.split('/')[-1]
            print('{0}/{1}: {2}'.format(str(index).zfill(len(str(url_count))),
                                        url_count,
                                        filename))
            
            temp_list = list()
            temp_list.append(url)
            
            d = Process(target = nsidc_download_tools.cmr_download ,args = (temp_list, dataDir+'/MODGRNLD/',))
            d.start()
            downloads.append(d)
        for dl in downloads:
            dl.join()
        ###########################################

        # Move downloaded files to the data directory
        #for url in MODGRNLD_url_list_download:
        #    filename = url.split('/')[-1]
        #    try:
        #        shutil.move(filename, dataDir + '/MODGRNLD/' + filename)
        #    except:
        #        print('')
        #        display(Markdown('<span style=''color:red''>Unexpected error: ' + str(sys.exc_info()[0]) + '</span>'))
        #        display(Markdown('<span style=''color:red''><b>ERROR in Moving Files to Data Directory! Data Directory may be have run out of space. Contact your administrator.</b></span>'))
        #        for url in MODGRNLD_url_list:
        #            filename = url.split('/')[-1]
        #            if os.path.isfile(filename):
        #                os.remove(filename)
        #                display(Markdown('<span style=''color:red''>File removed: ' + filename + '</span>'))
        #        return 1

    # Try opening the MODGRNLD files - this is a way of ensuring that they have been downloaded correctly. If
    # they cannot be opened for any reason, they will be deleted and the user is prompted to re-run the cell.
    try:
        ncfilename = [url.split('/')[-1] for url in MODGRNLD_url_list if url.endswith('.nc')][0]
        ds = Dataset(dataDir + '/MODGRNLD/' + ncfilename)
        ds.close()
    except:
        print('')
        display(Markdown('<span style=''color:red''>Unexpected error: ' + str(sys.exc_info()[0]) + '</span>'))
        display(Markdown('<span style=''color:red''><b>ERROR in File Download. MODGRNLD files will be deleted. \nFiles may not have been found or downloaded files were corrupted.</b></span>'))
        for url in MODGRNLD_url_list:
            filename = url.split('/')[-1]
            if os.path.isfile(dataDir+'/MODGRNLD/'+filename):
                os.remove(dataDir + '/MODGRNLD/' + filename)
                display(Markdown('<span style=''color:red''>File removed: ' + dataDir + '/MODGRNLD/' + filename + '</span>'))
        return 1
    return 0
                

# Download MERRA-2 data from GES DISC               
def download_merra_data():
    MERRA_url_list_download = list()
    ndays = calendar.monthrange(comparison_year, comparison_month)[1]
    
    if comparison_scope=='Monthly':
        MERRA_url = MERRA_url_template.format(comparison_year)
        MERRA_filename = MERRA_filename_template.format(file_number,comparison_year,comparison_month)
        job_output_file = open('job_info.txt','a')
        job_output_file.write(MERRA_filename+'\n')
        if os.path.exists(dataDir+'/MERRA-2/'+MERRA_filename):
            print('{:60s} already downloaded'.format(MERRA_filename))
        else:
            MERRA_url_list_download.append(MERRA_url+'/'+MERRA_filename)
    elif comparison_scope=='Daily':
        MERRA_url = MERRA_url_template.format(comparison_year, comparison_month)
        for day in range(1,ndays+1):
            MERRA_filename = MERRA_filename_template.format(file_number, comparison_year, comparison_month, day)
            job_output_file = open('job_info.txt','a')
            job_output_file.write(MERRA_filename+'\n')
            if os.path.exists(dataDir + '/MERRA-2/' + MERRA_filename):
                print('{:60s} already downloaded'.format(MERRA_filename))
            else:
                MERRA_url_list_download.append(MERRA_url + '/' + MERRA_filename)

    print('')
    if len(MERRA_url_list_download) > 0:
        # Download the files to the current directory
        #nsidc_download_tools.cmr_download(MERRA_url_list_download)
        #^Original Line
        
        ##Parallel Downloads##
        #only necessary for daily merra data#
        ###########################################
        url_count = len(MERRA_url_list_download)
        print('Downloading {0} files...'.format(url_count))
        
        temp_list = list()
        downloads = list()
        for index, url in enumerate(MERRA_url_list_download, start = 1):
            # Download the files to the current directory
            filename = url.split('/')[-1]
            print('{0}/{1}: {2}'.format(str(index).zfill(len(str(url_count))),
                                        url_count,
                                        filename))
            
            if comparison_variable=='Albedo':
                if index%5==0:
                    temp_list = list()
                    temp_list.append(url)
                    if index==url_count:
                        d = Process(target = nsidc_download_tools.cmr_download ,args = (temp_list, dataDir+'/MERRA-2/',))
                        d.start()
                        downloads.append(d)
                    else:
                        continue
                elif index%5==4:
                    temp_list.append(url)
                    d = Process(target = nsidc_download_tools.cmr_download ,args = (temp_list, dataDir+'/MERRA-2/',))
                    d.start()
                    downloads.append(d)
                else:
                    temp_list.append(url)
                    if index==url_count:
                        d = Process(target = nsidc_download_tools.cmr_download ,args = (temp_list, dataDir+'/MERRA-2/',))
                        d.start()
                        downloads.append(d)
                    else:
                        continue
            else:
                temp_list = list()
                temp_list.append(url)
                    
                d = Process(target = nsidc_download_tools.cmr_download ,args = (temp_list, dataDir+'/MERRA-2/',))
                d.start()
                downloads.append(d)
        
        for dl in downloads:
            dl.join()
        ###########################################
    
        # Move downloaded files to the data directory
        #moved = list()
        #for url in MERRA_url_list_download:
        #    MERRA_filename = url.split('/')[-1]
        #    try:
        #        d = Process(target = shutil.move, args = (MERRA_filename, dataDir+'/MERRA-2/'+MERRA_filename,))
        #        d.start()
        #        moved.append(d)
                #OG LINE
                #shutil.move(MERRA_filename, dataDir + '/MERRA-2/' + MERRA_filename)
        #    except:
        #        print('')
        #        display(Markdown('<span style=''color:red''>Unexpected error: ' + str(sys.exc_info()[0]) + '</span>'))
        #        display(Markdown('<span style=''color:red''><b>ERROR in Moving Files to Data Directory! Data Directory may be have run out of space. Contact your administrator.</b></span>'))
        #        for day in range(1,ndays+1):
        #            MERRA_filename = MERRA_filename_template.format(file_number, comparison_year, comparison_month, day)
        #            if os.path.isfile(MERRA_filename):
        #                os.remove(MERRA_filename)
        #                display(Markdown('<span style=''color:red''>File removed: ' + MERRA_filename + '</span>'))
        #        return 1
        #for mv in moved:
        #    mv.join()
        
    # Try opening the MERRA files - this is a way of ensuring that they have been downloaded correctly. If
    # they cannot be opened for any reason, they will be deleted and the user is prompted to re-run the cell.
    try:
        for day in range(1,ndays+1):
            MERRA_filename = MERRA_filename_template.format(file_number, comparison_year, comparison_month, day)
            MERRA_ncfile = dataDir + '/MERRA-2/' + MERRA_filename
            ds = Dataset(MERRA_ncfile)
            ds.close()
    except:
        #temp to stop files from being removed
        return 1
        ######################################
        print('')
        display(Markdown('<span style=''color:red''>Unexpected error: ' + str(sys.exc_info()[0]) + '</span>'))
        display(Markdown('<span style=''color:red''><b>ERROR in File Download. MERRA-2 files will be deleted. Files may have been corrupted or /data ran out of space.</b></span>'))
        for day in range(1,ndays+1):
            MERRA_filename = MERRA_filename_template.format(file_number, comparison_year, comparison_month, day)
            MERRA_ncfile = dataDir + '/MERRA-2/' + MERRA_filename
            if os.path.isfile(MERRA_ncfile):
                os.remove(MERRA_ncfile)
                display(Markdown('<span style=''color:red''>File removed: ' + dataDir + '/MERRA-2/' + MERRA_filename + '</span>'))
        return 1
    return 0
                
                
#   Read Greenland Ice Sheet mask             
def read_grnld_icesheet_mask():
    global mask, extent_mask
    # GrIS ice mask
    bedMachine_mask = dataDir + '/BedMachineGreenland-2017-09-20-Mask.tif.nc'
    ds = Dataset(bedMachine_mask)
    mask = ds['Band1'][:,:]
    x_mask = ds['x'][:]
    y_mask = ds['y'][:]
    ds.close()

    extent_mask = [x_mask[ 0]-(x_mask[ 1]-x_mask[ 0])/2, \
                   x_mask[-1]+(x_mask[ 1]-x_mask[ 0])/2, \
                   y_mask[-1]-(y_mask[-2]-y_mask[-1])/2, \
                   y_mask[ 0]+(y_mask[-2]-y_mask[-1])/2]

    if debug_plots:
        #plt.imshow(mask[::10,::10], extent=extent_mask, cmap=cm, vmin=0, vmax=5)
        plt.imshow(mask[::10,::10], extent=extent_mask, vmin=0, vmax=5)
        plt.colorbar()
        plt.gca().invert_yaxis()
    
    
#Read MERRA-2 data    
def read_merra_data():
    global Var_MERRA_mean, lon_MERRA, lat_MERRA
    # MERRA-2
    Var_MERRA_sum = None

    if comparison_scope=='Daily':
        ndays = calendar.monthrange(comparison_year, comparison_month)[1]
        for day in range(1,ndays+1):
            MERRA_filename = MERRA_filename_template.format(file_number, comparison_year, comparison_month, day)
            MERRA_ncfile = dataDir + '/MERRA-2/' + MERRA_filename
            if os.path.isfile(MERRA_ncfile):
                ds = Dataset(MERRA_ncfile)
            else:
                print(MERRA_ncfile+' not found! Skipping file. Data directory may have run out of space for all files.')
                continue
            #data subsetted based on user variable selection
            if comparison_variable=='Surface Temperature':
                Var_MERRA = ds['T2MMEAN'][:,:,:]
            elif comparison_variable=='Albedo':
                Var_MERRA = ds['ALBEDO'][:,:,:]
            elif comparison_variable=='Water Vapor':
                Var_MERRA = ds['TQV'][:,:,:]
    
            if day == 1:
                lat_MERRA = ds['lat'][:]
                lon_MERRA = ds['lon'][:]
                epsg_MERRA = 4326

            ds.close()
        
            # Daily average
            Var_MERRA_daily_avg = np.mean(Var_MERRA, axis=0)

            # Sum the variable
            if Var_MERRA_sum is None:
                Var_MERRA_sum = Var_MERRA_daily_avg
            else:
                Var_MERRA_sum = Var_MERRA_sum + Var_MERRA_daily_avg
               
        # Calculate the monthly mean
        #CHECK THIS AGAIN LATER IF ERROR THROWN
        Var_MERRA_mean = Var_MERRA_sum / ndays        
        
    elif comparison_scope=='Monthly':
        ndays = calendar.monthrange(comparison_year, comparison_month)[1]
        MERRA_filename = MERRA_filename_template.format(file_number, comparison_year, comparison_month)
        MERRA_ncfile = dataDir + '/MERRA-2/' + MERRA_filename
        if os.path.isfile(MERRA_ncfile):
            ds = Dataset(MERRA_ncfile)
        else:
            print(MERRA_ncfile+' not found! Skipping file. Data directory may have run out of space for all files.')
            return 1
        
        #data subsetted based on user variable selection
        if comparison_variable=='Surface Temperature':
            Var_MERRA = ds['T2MMEAN'][:,:,:]
        elif comparison_variable=='Albedo':
            Var_MERRA = ds['ALBEDO'][:,:,:]
        elif comparison_variable=='Water Vapor':
            Var_MERRA = ds['TQV'][:,:,:]
        
        lat_MERRA = ds['lat'][:]
        lon_MERRA = ds['lon'][:]
        epsg_MERRA = 4326
        
        ds.close()
        
        Var_MERRA_mean = np.mean(Var_MERRA, axis=0)
        
    # Select points over Greenland
    lat_min =  55.
    lat_max =  85.
    lon_min = -95.
    lon_max =  -5.
    row_min = np.where(lat_MERRA == lat_min)[0][0]
    row_max = np.where(lat_MERRA == lat_max)[0][0]
    col_min = np.where(lon_MERRA == lon_min)[0][0]
    col_max = np.where(lon_MERRA == lon_max)[0][0]

    Var_MERRA_mean = Var_MERRA_mean[row_min:row_max, col_min:col_max]
    lat_MERRA = lat_MERRA[row_min:row_max]
    lon_MERRA = lon_MERRA[col_min:col_max]
    extent_MERRA = [lon_min - (lon_MERRA[1]-lon_MERRA[0])/2, \
                    lon_max + (lon_MERRA[1]-lon_MERRA[0])/2, \
                    lat_min - (lat_MERRA[1]-lat_MERRA[0])/2, \
                    lat_max + (lat_MERRA[1]-lat_MERRA[0])/2]
   
    if comparison_variable=='Water Vapor':
        Var_MERRA_mean = Var_MERRA_mean * 0.01

    if debug_plots:
        plt.imshow(Var_MERRA_mean, origin='lower', extent=extent_MERRA)
        plt.colorbar()
        
    return 0
        
        
#  Read MODGRNLD data      
def read_modgrnld_data():
    global Var_MODIS_mean, extent_MODIS
    # MODGRNLD
    ulx, uly = -675000.0, -575000.0 # from user's guide on NSIDC
    lrx, lry = 887500.0, -3387500.0
    res = 781.25 # grid resolution
    x_MODIS = np.arange(ulx,lrx,res)
    y_MODIS = np.arange(lry,uly,res)
    extent_MODIS = [ulx-res/2, lrx+res/2, lry-res/2, uly+res/2]
    epsg_MODIS = 3411

    # Calculate monthly average
    ncfilename = [url.split('/')[-1] for url in MODGRNLD_url_list if url.endswith('.nc')][0]

    if os.path.isfile(dataDir+'/MODGRNLD/'+ncfilename):
        ds = Dataset(dataDir + '/MODGRNLD/' + ncfilename)
    else:
        display(Markdown('<span style=''color:red''>MODGRNLD file was not downloaded correctly. Data directory may have run out of space, or file may have been corrupted.' + dataDir + '/MERRA-2/' + ncfilename + '</span>'))
        return 1
    if comparison_variable=='Surface Temperature':
        Var_MODIS_mean = ds['Ice_Surface_Temperature_Mean'][0,:,:]
    elif comparison_variable=='Albedo':
        Var_MODIS_mean = ds['Albedo'][0,:,:].astype(float)
    elif comparison_variable=='Water Vapor':
        Var_MODIS_mean = ds['Water_Vapor_Near_Infrared_Mean'][0,:,:]
    ds.close()

    #MAY NEED TO ADD WATER VAPOR TO SURFACE TEMP OPTION
    #!!! COME BACK TO THIS
    if comparison_variable=='Surface Temperature' or comparison_variable=='Water Vapor':
        Var_MODIS_mean[Var_MODIS_mean== 0.] = np.nan
        Var_MODIS_mean[Var_MODIS_mean==50.] = np.nan
    elif comparison_variable=='Albedo':
        threshold = 100
        Var_MODIS_mean[Var_MODIS_mean > threshold] = np.nan
        Var_MODIS_mean = Var_MODIS_mean/100.
        
    if debug_plots:
        if comparison_variable=='Surface Temperature':
            plt.imshow(Var_MODIS_mean[::100,::100], extent=extent_MODIS)
            plt.colorbar()
        elif comparison_variable=='Albedo' or comparison_variabl=='Water Vapor':
            plt.imshow(Var_MODIS_mean[::10,::10], extent=extent_MODIS)
            plt.colorbar()
    
    return 0

        
        
def plot_modgrnld_data():
    # Ice_Surface_Temperature_Mean_Ndays
    ds = Dataset(dataDir + '/MODGRNLD/' + ncfilename)
    if comparison_variable=='Surface Temperature':
        MODGRNLD_Ndays = ds['Ice_Surface_Temperature_Mean_Ndays'][0,:,:]
    elif comparison_variable=='Albedo':
        MODGRNLD_Ndays = ds['Albedo_Ndays'][0,:,:]
    elif comparison_variable=='Water Vapor':
        MODGRNLD_Ndays = ds['Water_Vapor_Near_Infrared_Mean_Ndays'][0,:,:]
    ds.close()

    plt.figure(num=None, figsize=(10, 8))
    plt.imshow(MODGRNLD_Ndays[::1,::1], extent=[e/1000. for e in extent_MODIS])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('# of days')
    plt.title('number of days used in monthly average')
    plt.xlabel('km')
    plt.ylabel('km')
    
    plot_filename = 'MODGRNLD_data_map.pdf'
    plt.savefig(plot_filename)
    
    display(Download(plot_filename, label='Download plot'))
    plt.show()
            
        
#   Interpolate MERRA-2 and MODGRNLD data to a common grid¶    
def interpolate_to_common_grid():
    global Var_diff, Var_MODIS_interp
    # Project MERRA-2 values onto the MODIS polar stereographic projection
    lonm_MERRA, latm_MERRA = np.meshgrid(lon_MERRA, lat_MERRA)
    if pyproj.__version__[0] == '1':
        lonm_MERRA, latm_MERRA = np.meshgrid(lon_MERRA, lat_MERRA)
        xm_MERRA, ym_MERRA = transform(Proj(init="epsg:4326"), Proj(init="epsg:3411"), lonm_MERRA, latm_MERRA)
    elif pyproj.__version__[0] == '2' or pyproj.__version__[0] == '3':
        transformer = Transformer.from_crs("epsg:4326", "epsg:3411")
        xm_MERRA, ym_MERRA = transformer.transform(latm_MERRA, lonm_MERRA)


    # Bilinearly interpolate MODIS grid to MERRA-2 points
    gt_MODIS = raster.extent2gt(Var_MODIS_mean, extent_MODIS)
    i_MERRA, j_MERRA = raster.map2pixel(xm_MERRA, ym_MERRA, gt_MODIS)
    Var_MODIS_interp = bilinear_interpolate.bilinear_interpolate(Var_MODIS_mean, i_MERRA, j_MERRA)

    # Calculate the differences
    Var_diff = Var_MODIS_interp - Var_MERRA_mean

    if debug_plots:
        ##recheck this later with the ALbedo code
        fig, ax = plt.subplots(1,3)
        im2 = ax[0].scatter(xm_MERRA, ym_MERRA, c=Var_MERRA_mean,   vmin=210, vmax=270)
        im1 = ax[1].scatter(xm_MERRA, ym_MERRA, c=Var_MODIS_interp, vmin=210, vmax=270)
        im3 = ax[2].scatter(xm_MERRA, ym_MERRA, c=Var_diff, cmap='RdBu', vmin=-10, vmax=+10)

        ax[0].set_title('MERRA Var')
        ax[1].set_title('MODIS Var\n(interpolated)')
        ax[2].set_title('difference')
        
        
#  Plot MERRA-2 and MODGRNLD data and ice sheet surface temperature difference      
def plot_surfacetemp_diff():
    global Var_diff_masked, Var_MODIS_masked, Var_MERRA_masked
    # Project MERRA-2 values onto the BedMachine polar stereographic projection
    lonm_MERRA, latm_MERRA = np.meshgrid(lon_MERRA, lat_MERRA)
    if pyproj.__version__[0] == '1':
        lonm_MERRA, latm_MERRA = np.meshgrid(lon_MERRA, lat_MERRA)
        xm_MERRA, ym_MERRA = transform(Proj(init="epsg:4326"), Proj(init="epsg:3413"), lonm_MERRA, latm_MERRA)
    elif pyproj.__version__[0] == '2' or pyproj.__version__[0] == '3':
        transformer = Transformer.from_crs("epsg:4326", "epsg:3411")
        xm_MERRA, ym_MERRA = transformer.transform(latm_MERRA, lonm_MERRA)

    # Bilinearly interpolate BedMachine grid to MERRA-2 points
    gt_mask = raster.extent2gt(mask, extent_mask)
    i_MERRA, j_MERRA = raster.map2pixel(xm_MERRA, ym_MERRA, gt_mask)
    mask_interp = bilinear_interpolate.bilinear_interpolate(mask, i_MERRA, j_MERRA)
    ice_interp = mask_interp==2
    
    # Mask out points off ice
    if comparison_variable=='Surface Temperature' or comparison_variable=='Water Vapor' or comparison_variable=='Albedo':
        Var_MERRA_masked = Var_MERRA_mean
        Var_MERRA_masked[np.logical_not(ice_interp)] = np.nan
        Var_MODIS_masked = Var_MODIS_interp
        Var_MODIS_masked[np.logical_not(ice_interp)] = np.nan
        Var_diff_masked = Var_diff
        Var_diff_masked[np.logical_not(ice_interp)] = np.nan
    elif comparison_variable=='Albed':
        Var_MERRA_masked = Var_MERRA_mean
        Var_MERRA_masked[np.logical_not(ice_interp)] = math.isnan(np.nan)
        Var_MODIS_masked = Var_MODIS_interp
        Var_MODIS_masked[np.logical_not(ice_interp)] = math.isnan(np.nan)
        Var_diff_masked = Var_diff
        Var_diff_masked[np.logical_not(ice_interp)] = math.isnan(np.nan)
        
    cm = LinearSegmentedColormap.from_list('mask', mask_colors, N=5)
    
    if comparison_variable=='Surface Temperature' or comparison_variable=='Water Vapor' or comparison_variable=='Albedo':
        fig, ax = plt.subplots(1, 3, figsize=(16, 10))
    elif comparison_variable=='Albed':
        fig, ax = plt.subplots(1, 3, figsize=(24, 14))
    ax[0].imshow(mask[::10,::10], extent=extent_mask, cmap=cm, vmin=0, vmax=5)
    im1 = ax[0].scatter(xm_MERRA, ym_MERRA, c=Var_MERRA_mean)
    ax[0].invert_yaxis()
    ax[1].imshow(mask[::10,::10], extent=extent_mask, cmap=cm, vmin=0, vmax=5)
    im2 = ax[1].scatter(xm_MERRA, ym_MERRA, c=Var_MODIS_interp, vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
    ax[1].invert_yaxis()
    ax[2].imshow(mask[::10,::10], extent=extent_mask, cmap=cm, vmin=0, vmax=5)
    if comparison_variable=='Surface Temperature': #or comparison_variable=='Water Vapor':# or comparison_variable=='Albedo':
        im3 = ax[2].scatter(xm_MERRA, ym_MERRA, c=Var_diff, cmap=plt.cm.get_cmap('RdBu').reversed(), vmin=-10, vmax=+10)
    elif comparison_variable=='Water Vapor' or comparison_variable=='Albedo':
        im3 = ax[2].scatter(xm_MERRA, ym_MERRA, c=Var_diff, cmap=plt.cm.get_cmap('RdBu').reversed(), vmin=-0.20, vmax=+0.20)
    ax[2].invert_yaxis()

    cbar1 = fig.colorbar(im1, ax=ax[0], orientation='horizontal')
    cbar2 = fig.colorbar(im2, ax=ax[1], orientation='horizontal')
    cbar3 = fig.colorbar(im3, ax=ax[2], orientation='horizontal')
    
    if comparison_variable=='Surface Temperature':
        cbar1.ax.set_xlabel('temperature (K)')
        cbar2.ax.set_xlabel('temperature (K)')
        cbar3.ax.set_xlabel('temperature (K)')
        ax[0].set_title('MERRA Ts')
        ax[1].set_title('MODIS Ts\n(interpolated)')
        ax[2].set_title('difference')
    elif comparison_variable=='Albedo':
        cbar1.ax.set_xlabel('Albedo (%)')
        cbar2.ax.set_xlabel('Albedo (%)')
        cbar3.ax.set_xlabel('Albedo difference (%)')
        ax[0].set_title('MERRA Albedo')
        ax[1].set_title('MODIS Albedo\n(interpolated)')
        ax[2].set_title('difference')
    elif comparison_variable=='Water Vapor':
        cbar1.ax.set_xlabel('water vapor (cm)')
        cbar2.ax.set_xlabel('water vapor (cm)')
        cbar3.ax.set_xlabel('water vapor (cm)')
        ax[0].set_title('MERRA WV')
        ax[1].set_title('MODIS WV\n(interpolated)')
        ax[2].set_title('difference')
        
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

    plot_filename = 'MODGRNLD_MERRA_MODIS_Var_comparison_map.pdf'
    plt.savefig(plot_filename)
    
    display(Download(plot_filename, label='Download plot'))
    plt.show()
        
#   Calculate and plot statistics        
def calc_and_plot_stats():
    # Fix array size
    Var_MODIS_masked_nonan = Var_MODIS_masked[np.logical_not(np.isnan(Var_MODIS_masked))]
    Var_MERRA_masked_nonan = Var_MERRA_masked[np.logical_not(np.isnan(Var_MERRA_masked))]
    
    # Convert tuple to int
    size = int(''.join(map(str, Var_MODIS_masked_nonan.shape)))
    data = Var_MODIS_masked_nonan
    xloc = np.arange(size)
    newsize = int(''.join(map(str, Var_MERRA_masked_nonan.shape)))
    new_xloc = np.linspace(0, size, newsize)
    new_data = np.interp(new_xloc, xloc, data)

    Var_MODIS_masked_nonan_int = new_data
    
    # Remove the nans (off-ice points)
    Var_diff_masked_nonan = Var_diff_masked[np.logical_not(np.isnan(Var_diff_masked))]

    # Histogram
    n, bins, patches = plt.hist(Var_diff_masked_nonan, bins=100)
    if comparison_variable=='Surface Temperature':
        plt.xlabel('Temperature Difference (MODIS minus MERRA) [K]')
    elif comparison_variable=='Albedo':
        plt.xlabel('Albedo Difference (MODIS minus MERRA)')
    plt.ylabel('count')
    
    # Statistics
    Var_diff_mean = np.mean(Var_diff_masked_nonan)
    Var_diff_stdv = np.std(Var_diff_masked_nonan)
    Var_diff_corr = np.corrcoef(Var_diff_masked_nonan)
    
    #Calculate Pearson's Correlation
    corr, _ =pearsonr(Var_MODIS_masked_nonan_int, Var_MERRA_masked_nonan)
    
    mse = mean_squared_error(Var_MODIS_masked_nonan_int, Var_MERRA_masked_nonan , squared=False)
    rmse = sqrt(mse)

    plot_filename = 'MODGRNLD_MERRA_MODIS_Var_comparison_histogram.pdf'
    plt.savefig(plot_filename)
    
    print('MODIS [minus] MERRA-2 statistics:')
    if comparison_variable=='Surface Temperature':
        print(' mean = {:+6.2f} K'.format(Var_diff_mean))
        print(' stdv = {:+6.2f} K'.format(Var_diff_stdv))
    elif comparison_variable=='Albedo':
        print(' mean = {:+6.2f}'.format(Var_diff_mean))
        print(' stdv = {:+6.2f}'.format(Var_diff_stdv))
    elif comparison_variable=='Water Vapor':
        print(' mean = {:+6.2f} cm'.format(Var_diff_mean))
        print(' stdv = {:+6.2f} cm'.format(Var_diff_stdv))
    print(' Pearsons correlation = {:+6.2f}'.format(corr))
    print(' RMSE = {:+6.2f}'.format(rmse))
    display(Download(plot_filename, label='Download plot'))
    plt.show()
    
def clean_up():
    for url in MODGRNLD_url_list:
        MODGRNLD_filename = url.split('/')[-1]
        MODGRNLD_ncfile = dataDir + '/MODGRNLD/' + MODGRNLD_filename
        if os.path.exists(MODGRNLD_ncfile):
            os.remove(MODGRNLD_ncfile)
    #display(Markdown('<span style=''color:red''>MODGRNLD Files removed! They occupy too much space in /data. '+'</span>'))
    
    ndays = calendar.monthrange(comparison_year, comparison_month)[1]
    for day in range(1,ndays+1):
        MERRA_filename = MERRA_filename_template.format(file_number, comparison_year, comparison_month, day)
        MERRA_ncfile = dataDir + '/MERRA-2/' + MERRA_filename
        if os.path.isfile(MERRA_ncfile):
            os.remove(MERRA_ncfile)
    #display(Markdown('<span style=''color:red''>MERRA-2 Files removed! They occupy too much space in /data. '+'</span>'))