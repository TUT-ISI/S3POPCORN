import numpy as np
import sys
import os
from glob import glob
from datetime import datetime
import joblib
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from netCDF4 import Dataset
from zipfile import ZipFile
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d
from sklearn.neighbors import KNeighborsRegressor
import time
from sklearn.utils import parallel_backend

# #########################################################################
# POPCORN Sentinel-3 Synergy aerosol parameter post-process correction
# CODE VERSION 27 Aug 2021.
#
#   * Developed by: Finnish Meteorological Institute and University of Eastern Finland
#   * Development of the algorithm was funded by the European Space Agency
#     EO science for society programme via POPCORN project.
#   * Contact info: Antti Lipponen (antti.lipponen@fmi.fi)
#
# #########################################################################


# #########################################################################
# Function to parse bitmasks
# #########################################################################
def parseBitMasks(flag_masks, flag_meanings, data):
    mask = {}
    for ii in range(len(flag_masks)):
        mask[flag_meanings[ii]] = np.bitwise_and(data, flag_masks[ii]) > 0
    return mask


# #########################################################################
# Compute the scattering angle given the solar zenith angle (sza), view zenith angle (vza), and relative azimuth angle (relAz)
# #########################################################################
def computeScatteringAngle(sza, vza, relAz):
    return np.rad2deg(np.arccos(np.clip(-np.cos(np.deg2rad(sza)) * np.cos(np.deg2rad(vza)) + np.sin(np.deg2rad(sza)) * np.sin(np.deg2rad(vza)) * np.cos(np.deg2rad(relAz)), a_min=-1, a_max=1)))


# #########################################################################
# Load Sentinel-3 data
# #########################################################################
def loadS3_SY_OL_SL(SYfilename, OLfilename, SLfilename, n_jobs=8):
    try:
        t0 = time.time()
        print('Loading data.')
        print(SYfilename)
        print(OLfilename)
        print(SLfilename)

        data = {}

        # #########################################################################################################################
        # SYN
        # #########################################################################################################################
        print('\n  SYN...')
        archive = ZipFile(SYfilename, 'r')

        # time.nc
        print('    time.nc')
        ncfilename = os.path.join(os.path.split(SYfilename)[-1].replace('zip', 'SEN3'), 'time.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            data['start_time_str'] = nc.start_time  # '%Y-%m-%dT%H:%M:%S.%fZ'
            data['stop_time_str'] = nc.stop_time  # '%Y-%m-%dT%H:%M:%S.%fZ'
            if nc.start_time[:4] != '2019':
                print('    ************************************')
                print('    *** Warning: Training of the POPCORN models was carried out using Sentinel-3 data from year 2019 - this file has data for year {}'.format(nc.start_time[:4]))
                print('    ************************************')
            if not (nc.source == 'IPF-SY-2 06.16' or nc.source == 'IPF-SY-2 06.17'):
                print('    ************************************')
                print('    *** Warning: Training of the POPCORN models was carried out using Synergy data version IPF-SY-2 06.16 and IPF-SY-2 06.17 - this file has version {}'.format(nc.source))
                print('    ************************************')
        # geolocation.nc
        print('    geolocation.nc')
        geolocationncfilename = os.path.join(os.path.split(SYfilename)[-1].replace('zip', 'SEN3'), 'geolocation.nc')
        with Dataset('dummy', mode='r', memory=archive.open(geolocationncfilename).read()) as nc:
            data['altitude'] = nc['altitude'][:]
            data['lat'] = nc['lat'][:]
            data['lon'] = nc['lon'][:]
        latmax, latmin, latmean = data['lat'].max(), data['lat'].min(), data['lat'].mean()
        lonmax, lonmin, lonmean = data['lon'].max(), data['lon'].min(), data['lon'].mean()

        # POPCORN regions of interest
        ROIs = {
            'Central Europe': {'latmin': 39, 'latmax': 54, 'lonmin': -8, 'lonmax': 20},
            'Western USA': {'latmin': 35, 'latmax': 50, 'lonmin': -125, 'lonmax': -115},
            'Eastern USA': {'latmin': 27, 'latmax': 45, 'lonmin': -84, 'lonmax': -74},
            'Southern Africa': {'latmin': -35.5, 'latmax': -12.5, 'lonmin': 12, 'lonmax': 35},
            'India': {'latmin': 8.5, 'latmax': 31.5, 'lonmin': 70, 'lonmax': 92}
        }

        insideROI = None
        for ROI in ROIs:
            if np.logical_and.reduce((ROIs[ROI]['latmin'] <= latmean, ROIs[ROI]['latmax'] >= latmean, ROIs[ROI]['lonmin'] <= lonmean, ROIs[ROI]['lonmax'] >= lonmean)):
                insideROI = ROI

        if insideROI is None:
            print('    ************************************')
            print('    *** Warning: Training of the POPCORN models was carried out using Synergy data from five regions of interest. You are most likely evaluating data outside these regions.')
            print('    ************************************')
        else:
            print('      Data inside the POPCORN region of interest: {}'.format(insideROI))

        # Syn_AMIN.nc
        print('    Syn_AMIN.nc')
        ncfilename = os.path.join(os.path.split(SYfilename)[-1].replace('zip', 'SEN3'), 'Syn_AMIN.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            data['SYN_AMIN'] = nc['AMIN'][:]

        # Syn_Angstrom_exp550.nc
        print('    Syn_Angstrom_exp550.nc')
        ncfilename = os.path.join(os.path.split(SYfilename)[-1].replace('zip', 'SEN3'), 'Syn_Angstrom_exp550.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            data['SYN_AE550'] = nc['A550'][:]

        # Syn_AOT550.nc
        print('    Syn_AOT550.nc')
        ncfilename = os.path.join(os.path.split(SYfilename)[-1].replace('zip', 'SEN3'), 'Syn_AOT550.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            data['SYN_AOD550'] = nc['T550'][:]
            data['SYN_AOD550err'] = nc['T550_err'][:]
            data['SYN_AOD550mask'] = ~nc['T550'][:].mask

        if data['SYN_AOD550mask'].sum() == 0:
            return None

        # flags.nc
        print('    flags.nc')
        ncfilename = os.path.join(os.path.split(SYfilename)[-1].replace('zip', 'SEN3'), 'flags.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            SYN_flags_meanings = nc['SYN_flags'].flag_meanings.split(' ')
            SYN_flags_masks = list(nc['SYN_flags'].flag_masks)
            SYN_flags = nc['SYN_flags'][:]
            data['SYN_flags'] = parseBitMasks(SYN_flags_masks, SYN_flags_meanings, SYN_flags.data)
            CLOUD_flags_meanings = nc['CLOUD_flags'].flag_meanings.split(' ')
            CLOUD_flags_masks = list(nc['CLOUD_flags'].flag_masks)
            CLOUD_flags = nc['CLOUD_flags'][:]
            data['CLOUD_flags'] = parseBitMasks(CLOUD_flags_masks, CLOUD_flags_meanings, CLOUD_flags.data)
            OLC_flags_meanings = nc['OLC_flags'].flag_meanings.split(' ')
            OLC_flags_masks = list(nc['OLC_flags'].flag_masks)
            OLC_flags = nc['OLC_flags'][:]
            data['OLC_flags'] = parseBitMasks(OLC_flags_masks, OLC_flags_meanings, OLC_flags.data)
            SLN_flags_meanings = nc['SLN_flags'].flag_meanings.split(' ')
            SLN_flags_masks = list(nc['SLN_flags'].flag_masks)
            SLN_flags = nc['SLN_flags'][:]
            data['SLN_flags'] = parseBitMasks(SLN_flags_masks, SLN_flags_meanings, SLN_flags.data)
            SLO_flags_meanings = nc['SLO_flags'].flag_meanings.split(' ')
            SLO_flags_masks = list(nc['SLO_flags'].flag_masks)
            SLO_flags = nc['SLO_flags'][:]
            data['SLO_flags'] = parseBitMasks(SLO_flags_masks, SLO_flags_meanings, SLO_flags.data)

        data['SYN_SYN_no_slo'] = data['SYN_flags']['SYN_no_slo']
        data['SYN_SYN_no_sln'] = data['SYN_flags']['SYN_no_sln']
        data['SYN_SYN_no_olc'] = data['SYN_flags']['SYN_no_olc']

        for k in data['SYN_flags']:
            data['SYN_' + k] = data['SYN_flags'][k]
        for k in data['CLOUD_flags']:
            data['SYN_' + k] = data['CLOUD_flags'][k]
        for k in data['OLC_flags']:
            data['SYN_' + k] = data['OLC_flags'][k]
        for k in data['SLN_flags']:
            data['SYN_' + k] = data['SLN_flags'][k]
        for k in data['SLO_flags']:
            data['SYN_' + k] = data['SLO_flags'][k]

        # Syn_Oa01_reflectance.nc
        # Syn_Oa02_reflectance.nc
        # Syn_Oa03_reflectance.nc
        # Syn_Oa04_reflectance.nc
        # Syn_Oa05_reflectance.nc
        # Syn_Oa06_reflectance.nc
        # Syn_Oa07_reflectance.nc
        # Syn_Oa08_reflectance.nc
        # Syn_Oa09_reflectance.nc
        # Syn_Oa10_reflectance.nc
        # Syn_Oa11_reflectance.nc
        # Syn_Oa12_reflectance.nc
        # Syn_Oa16_reflectance.nc
        # Syn_Oa17_reflectance.nc
        # Syn_Oa18_reflectance.nc
        # Syn_Oa21_reflectance.nc
        # Syn_S1N_reflectance.nc
        # Syn_S1O_reflectance.nc
        # Syn_S2N_reflectance.nc
        # Syn_S2O_reflectance.nc
        # Syn_S3N_reflectance.nc
        # Syn_S3O_reflectance.nc
        # Syn_S5N_reflectance.nc
        # Syn_S5O_reflectance.nc
        # Syn_S6N_reflectance.nc
        # Syn_S6O_reflectance.nc
        SDRvars = ['Oa01', 'Oa02', 'Oa03', 'Oa04', 'Oa05', 'Oa06', 'Oa07', 'Oa08', 'Oa09', 'Oa10', 'Oa11', 'Oa12', 'Oa16', 'Oa17', 'Oa18', 'Oa21', 'S1N', 'S1O', 'S2N', 'S2O', 'S3N', 'S3O', 'S5N', 'S5O', 'S6N', 'S6O']
        for SDR in SDRvars:
            print('    Syn_' + SDR + '_reflectance.nc')
            ncfilename = os.path.join(os.path.split(SYfilename)[-1].replace('zip', 'SEN3'), 'Syn_' + SDR + '_reflectance.nc')
            with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
                data['SYN_SDR_' + SDR] = nc['SDR_' + SDR][:]
                data['SYN_SDR_' + SDR][data['SYN_SDR_' + SDR].mask] = np.nan

        # tiepoints_olci.nc
        print('    tiepoints_olci.nc')
        ncfilename = os.path.join(os.path.split(SYfilename)[-1].replace('zip', 'SEN3'), 'tiepoints_olci.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            nc.set_auto_mask(False)
            OLC_TP_lat, OLC_TP_lon, OLC_VAA, OLC_VZA, SAA, SZA = nc['OLC_TP_lat'][:], nc['OLC_TP_lon'][:], nc['OLC_VAA'][:], nc['OLC_VZA'][:], nc['SAA'][:], nc['SZA'][:]

        tri = Delaunay(np.hstack((OLC_TP_lat[:, None], OLC_TP_lon[:, None])).data)
        OMETEO = LinearNDInterpolator(tri, np.hstack((OLC_VAA[:, None], OLC_VZA[:, None], SAA[:, None], SZA[:, None])))(data['lat'], data['lon'])
        data['SYN_O_VAA'] = OMETEO[:, :, 0]
        data['SYN_O_VZA'] = OMETEO[:, :, 1]
        data['SYN_O_SAA'] = OMETEO[:, :, 2]
        data['SYN_O_SZA'] = OMETEO[:, :, 3]
        OMETEO = None

        # tiepoints_slstr_n.nc
        print('    tiepoints_slstr_n.nc')
        ncfilename = os.path.join(os.path.split(SYfilename)[-1].replace('zip', 'SEN3'), 'tiepoints_slstr_n.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            nc.set_auto_mask(False)
            SLN_TP_lat, SLN_TP_lon, SLN_VAA, SLN_VZA = nc['SLN_TP_lat'][:], nc['SLN_TP_lon'][:], nc['SLN_VAA'][:], nc['SLN_VZA'][:]
        with parallel_backend('threading'):
            nearestpoints_tiepoints_slstr_n_CLF = KNeighborsRegressor(n_neighbors=1, leaf_size=10, n_jobs=n_jobs).fit(np.hstack((SLN_TP_lat[:, None], SLN_TP_lon[:, None])).data, np.zeros((len(SLN_TP_lat), 1)))
            indices_tiepoints_slstr_n = nearestpoints_tiepoints_slstr_n_CLF.kneighbors(np.hstack((data['lat'].ravel()[:, None], data['lon'].ravel()[:, None])), return_distance=False)
        nearestpoints_tiepoints_slstr_n_CLF = None  # Free memory
        data['SYN_SN_VAA'] = np.reshape(SLN_VAA[indices_tiepoints_slstr_n].ravel(), data['lat'].shape)
        data['SYN_SN_VZA'] = np.reshape(SLN_VZA[indices_tiepoints_slstr_n].ravel(), data['lat'].shape)
        SLN_TP_lat, SLN_TP_lon, SLN_VAA, SLN_VZA = None, None, None, None

        # tiepoints_slstr_o.nc
        print('    tiepoints_slstr_o.nc')
        ncfilename = os.path.join(os.path.split(SYfilename)[-1].replace('zip', 'SEN3'), 'tiepoints_slstr_o.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            nc.set_auto_mask(False)
            SLO_TP_lat, SLO_TP_lon, SLO_VAA, SLO_VZA = nc['SLO_TP_lat'][:], nc['SLO_TP_lon'][:], nc['SLO_VAA'][:], nc['SLO_VZA'][:]
        M = np.logical_and.reduce((SLO_TP_lat >= latmin - 0.5, SLO_TP_lat <= latmax + 0.5, SLO_TP_lon >= lonmin - 0.5, SLO_TP_lon <= lonmax + 0.5))
        SLO_TP_lat, SLO_TP_lon, SLO_VAA, SLO_VZA = SLO_TP_lat[M], SLO_TP_lon[M], SLO_VAA[M], SLO_VZA[M]
        with parallel_backend('threading'):
            nearestpoints_tiepoints_slstr_o_CLF = KNeighborsRegressor(n_neighbors=1, leaf_size=10, n_jobs=n_jobs).fit(np.hstack((SLO_TP_lat[:, None], SLO_TP_lon[:, None])).data, np.zeros((len(SLO_TP_lat), 1)))
            indices_tiepoints_slstr_o = nearestpoints_tiepoints_slstr_o_CLF.kneighbors(np.hstack((data['lat'].ravel()[:, None], data['lon'].ravel()[:, None])), return_distance=False)
        nearestpoints_tiepoints_slstr_o_CLF = None  # Free memory

        data['SYN_SO_VAA'] = np.reshape(SLO_VAA[indices_tiepoints_slstr_o].ravel(), data['lat'].shape)
        data['SYN_SO_VZA'] = np.reshape(SLO_VZA[indices_tiepoints_slstr_o].ravel(), data['lat'].shape)
        SLO_TP_lat, SLO_TP_lon, SLO_VAA, SLO_VZA = None, None, None, None

        archive.close()

        # #########################################################################################################################
        # OLCI
        # #########################################################################################################################
        print('  OLCI...')
        archive = ZipFile(OLfilename, 'r')

        print('    geo_coordinates.nc')
        ncfilename = os.path.join(os.path.split(OLfilename)[-1].replace('zip', 'SEN3'), 'geo_coordinates.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            nY = nc['latitude'].shape[1]
            lat = nc['latitude'][:]
            lon = nc['longitude'][:]
            if nc.source != 'IPF-OL-1-EO 06.08':
                print('    ************************************')
                print('    *** Warning: Training of the POPCORN models was carried out using OLCI level-1 data version IPF-OL-1-EO 06.08 - this file has version {}'.format(nc.source))
                print('    ************************************')

        MASK = np.logical_and.reduce((lat >= latmin - 0.05, lat <= latmax + 0.05, lon >= lonmin - 0.05, lon <= lonmax + 0.05))
        lat, lon = lat[MASK], lon[MASK]

        print('    qualityFlags.nc')
        ncfilename = os.path.join(os.path.split(OLfilename)[-1].replace('zip', 'SEN3'), 'qualityFlags.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            MASK_qf = parseBitMasks(nc['quality_flags'].flag_masks, nc['quality_flags'].flag_meanings.split(' '), nc['quality_flags'][:][MASK])

        print('    instrument_data.nc')
        ncfilename = os.path.join(os.path.split(OLfilename)[-1].replace('zip', 'SEN3'), 'instrument_data.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            detector_index = nc['detector_index'][:][MASK]
            solar_flux = nc['solar_flux'][:]

        print('    tie_geometries.nc')
        ncfilename = os.path.join(os.path.split(OLfilename)[-1].replace('zip', 'SEN3'), 'tie_geometries.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            SZA_tie = nc['SZA'][:]
            SZA = interp1d(np.linspace(0.0, 1.0, SZA_tie.shape[1]), SZA_tie, axis=1)(np.linspace(0.0, 1.0, nY))[MASK]

        rhoTOA = {}
        for b in np.arange(1, 21 + 1):
            print('    Oa{:02d}_radiance.nc'.format(b))
            ncfilename = os.path.join(os.path.split(OLfilename)[-1].replace('zip', 'SEN3'), 'Oa{:02d}_radiance.nc'.format(b))
            with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
                solar_irradiance = solar_flux[b - 1, :][detector_index]
                rhoTOA_thisband = np.array(np.pi * (nc['Oa{:02d}_radiance'.format(b)][:][MASK] / solar_irradiance) / np.cos(np.deg2rad(SZA)))
                rhoTOA_thisband[rhoTOA_thisband > 1.0] = np.nan
                rhoTOA['Oa{:02d}_reflectance'.format(b)] = rhoTOA_thisband
        rhoTOA_thisband = None

        with parallel_backend('threading'):
            nearestpointsCLF = KNeighborsRegressor(n_neighbors=1, leaf_size=10, n_jobs=n_jobs).fit(np.hstack((lat[:, None], lon[:, None])).data, np.zeros((len(lat), 1)))
            indices = nearestpointsCLF.kneighbors(np.hstack((data['lat'].ravel()[:, None], data['lon'].ravel()[:, None])), return_distance=False)
        nearestpointsCLF = None  # Free memory

        for k in rhoTOA:
            data['OL1_' + k] = np.reshape(rhoTOA[k][indices][:, 0], data['lat'].shape)
        for k in MASK_qf:
            data['OL1_flags_' + k] = np.array(np.reshape(MASK_qf[k][indices][:, 0], data['lat'].shape))
        rhoTOA, indices = None, None

        archive.close()

        # #########################################################################################################################
        # SLSTR
        # #########################################################################################################################
        print('  SLSTR...')
        archive = ZipFile(SLfilename, 'r')

        print('    geodetic_an.nc')
        ncfilename = os.path.join(os.path.split(SLfilename)[-1].replace('zip', 'SEN3'), 'geodetic_an.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            lat_an = nc['latitude_an'][:]
            lon_an = nc['longitude_an'][:]
            if nc.source != 'IPF-SL-1 06.16':
                print('    ************************************')
                print('    *** Warning: Training of the POPCORN models was carried out using SLSTR level-1 data version IPF-SL-1 06.16 - this file has version {}'.format(nc.source))
                print('    ************************************')

        print('    geodetic_ao.nc')
        ncfilename = os.path.join(os.path.split(SLfilename)[-1].replace('zip', 'SEN3'), 'geodetic_ao.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            lat_ao = nc['latitude_ao'][:]
            lon_ao = nc['longitude_ao'][:]

        print('    geodetic_tx.nc')
        ncfilename = os.path.join(os.path.split(SLfilename)[-1].replace('zip', 'SEN3'), 'geodetic_tx.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            lat_tx = nc['latitude_tx'][:]
            lon_tx = nc['longitude_tx'][:]

        print('    geometry_tn.nc')
        ncfilename = os.path.join(os.path.split(SLfilename)[-1].replace('zip', 'SEN3'), 'geometry_tn.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            sza_tn = nc['solar_zenith_tn'][:]

        print('    geometry_to.nc')
        ncfilename = os.path.join(os.path.split(SLfilename)[-1].replace('zip', 'SEN3'), 'geometry_to.nc')
        with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
            sza_to = nc['solar_zenith_to'][:]

        with parallel_backend('threading'):
            nearestpoints_SZA_CLF = KNeighborsRegressor(n_neighbors=1, leaf_size=10, n_jobs=n_jobs).fit(np.hstack((lat_tx.ravel()[:, None], lon_tx.ravel()[:, None])).data, np.zeros((len(lat_tx.ravel()), 1)))
            SZA_indices = nearestpoints_SZA_CLF.kneighbors(np.hstack((data['lat'].ravel()[:, None], data['lon'].ravel()[:, None])), return_distance=False)
        nearestpoints_SZA_CLF = None  # Free memory
        SZA_ao = sza_to.ravel()[SZA_indices].data
        SZA_an = sza_tn.ravel()[SZA_indices].data
        SZA_indices = None

        with parallel_backend('threading'):
            nearestpoints_ao_CLF = KNeighborsRegressor(n_neighbors=1, leaf_size=10, n_jobs=n_jobs).fit(np.hstack((lat_ao.ravel()[:, None], lon_ao.ravel()[:, None])).data, np.zeros((len(lat_ao.ravel()), 1)))
            indices_ao = nearestpoints_ao_CLF.kneighbors(np.hstack((data['lat'].ravel()[:, None], data['lon'].ravel()[:, None])), return_distance=False)
        nearestpoints_ao_CLF = None  # Free memory

        with parallel_backend('threading'):
            nearestpoints_an_CLF = KNeighborsRegressor(n_neighbors=1, leaf_size=10, n_jobs=n_jobs).fit(np.hstack((lat_an.ravel()[:, None], lon_an.ravel()[:, None])).data, np.zeros((len(lat_an.ravel()), 1)))
            indices_an = nearestpoints_an_CLF.kneighbors(np.hstack((data['lat'].ravel()[:, None], data['lon'].ravel()[:, None])), return_distance=False)
        nearestpoints_an_CLF = None  # Free memory
        lat_an, lon_an, lat_ao, lon_ao, lat_tx, lon_tx, sza_tn, sza_to = None, None, None, None, None, None, None, None

        for b in range(1, 6 + 1):
            print('    S{:d}_radiance_an.nc'.format(b))
            ncfilename = os.path.join(os.path.split(SLfilename)[-1].replace('zip', 'SEN3'), 'S{:d}_radiance_an.nc'.format(b))
            with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
                radiance_an = np.array(nc['S{:d}_radiance_an'.format(b)][:])
                radiance_an[radiance_an < 0] = np.nan
            print('    S{:d}_quality_an.nc'.format(b))
            ncfilename = os.path.join(os.path.split(SLfilename)[-1].replace('zip', 'SEN3'), 'S{:d}_quality_an.nc'.format(b))
            with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
                solar_irradiance_an = float(nc['S{:d}_solar_irradiance_an'.format(b)][0])
            ncfilename = os.path.join(os.path.split(SLfilename)[-1].replace('zip', 'SEN3'), 'S{:d}_radiance_ao.nc'.format(b))
            print('    S{:d}_radiance_ao.nc'.format(b))
            with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
                radiance_ao = np.array(nc['S{:d}_radiance_ao'.format(b)][:])
                radiance_ao[radiance_ao < 0] = np.nan
            ncfilename = os.path.join(os.path.split(SLfilename)[-1].replace('zip', 'SEN3'), 'S{:d}_quality_ao.nc'.format(b))
            print('    S{:d}_quality_ao.nc'.format(b))
            with Dataset('dummy', mode='r', memory=archive.open(ncfilename).read()) as nc:
                solar_irradiance_ao = float(nc['S{:d}_solar_irradiance_ao'.format(b)][0])
            TOA_reflectance_an = np.array(np.pi * (radiance_an.ravel()[indices_an] / solar_irradiance_an) / np.cos(np.deg2rad(SZA_an)))
            data['SL1_S{:d}_reflectance_nadir'.format(b)] = np.reshape(TOA_reflectance_an, data['lon'].shape)
            TOA_reflectance_ao = np.array(np.pi * (radiance_ao.ravel()[indices_ao] / solar_irradiance_ao) / np.cos(np.deg2rad(SZA_ao)))
            data['SL1_S{:d}_reflectance_oblique'.format(b)] = np.reshape(TOA_reflectance_ao, data['lon'].shape)
            radiance_an, solar_irradiance_an, radiance_ao, solar_irradiance_ao = None, None, None, None
        archive.close()

        print('Final computations...')
        vaa = data['SYN_O_VAA']
        saa = data['SYN_O_SAA']
        relAz = np.abs(saa - vaa - 180.0)
        relAz[relAz > 360.0] = np.mod(relAz[relAz > 360.0], 360.0)
        relAz[relAz > 180.0] = 360.0 - relAz[relAz > 180.0]
        data['SYN_O_scattering_angle'] = computeScatteringAngle(data['SYN_O_SZA'], data['SYN_O_VZA'], relAz)
        vaa = data['SYN_SO_VAA']
        saa = data['SYN_O_SAA']
        relAz = np.abs(saa - vaa - 180.0)
        relAz[relAz > 360.0] = np.mod(relAz[relAz > 360.0], 360.0)
        relAz[relAz > 180.0] = 360.0 - relAz[relAz > 180.0]
        data['SYN_SO_scattering_angle'] = computeScatteringAngle(data['SYN_O_SZA'], data['SYN_SO_VZA'], relAz)
        vaa = data['SYN_SN_VAA']
        saa = data['SYN_O_SAA']
        relAz = np.abs(saa - vaa - 180.0)
        relAz[relAz > 360.0] = np.mod(relAz[relAz > 360.0], 360.0)
        relAz[relAz > 180.0] = 360.0 - relAz[relAz > 180.0]
        data['SYN_SN_scattering_angle'] = computeScatteringAngle(data['SYN_O_SZA'], data['SYN_SN_VZA'], relAz)

        print('Done! (Duration: {:.2f} s)'.format(time.time() - t0))

    except Exception:
        return None

    return data


print('')
print('POPCORN Sentinel-3 Synergy aerosol parameter post-process correction')
print('CODE VERSION 27 Aug 2021.')
print('')
print('  Finnish Meteorological Institute and University of Eastern Finland')
print('  Development of the algorithm was funded by the European Space Agency EO science for society programme via POPCORN project.')
print('  Contact info: Antti Lipponen (antti.lipponen@fmi.fi)')
print('')

if len(sys.argv) != 4:
    print('USE: python S3POPCORN.py BASEDIR OUTPUTDIR SYNFILE')
    print('    BASEDIR=directory under which the Sentinel-3 data will be looked for (uses recursive search). For example ".".')
    print('    OUTPUTDIR=directory in which the post-process corrected data will be saver')
    print('    SYNFILE=name of the zip file that contains the Synergy data')
    print('')
    print('    Sentinel-3 level-1 and level-2 data products are expected to be stored in zip format.')
    sys.exit(1)

basedir = sys.argv[1]
OUTPUTDIR = sys.argv[2]
SYfilename = sys.argv[3]
SYfilename_original = sys.argv[3]

satellite = SYfilename[:3]
SYfilename = glob(os.path.join(basedir, '**', SYfilename), recursive=True)
if len(SYfilename) == 0:
    print('SYN file given not found')
    sys.exit(1)
if len(SYfilename) > 1:
    print('Multiple SYN files matching the name given found')
    sys.exit(1)
SYfilename = SYfilename[0]

SYtimes = os.path.split(SYfilename)[-1][16:47].split('_')
SYt0 = datetime(int(SYtimes[0][:4]), int(SYtimes[0][4:6]), int(SYtimes[0][6:8]), int(SYtimes[0][9:11]), int(SYtimes[0][11:13]), int(SYtimes[0][13:15]))
SYt1 = datetime(int(SYtimes[1][:4]), int(SYtimes[1][4:6]), int(SYtimes[1][6:8]), int(SYtimes[1][9:11]), int(SYtimes[1][11:13]), int(SYtimes[1][13:15]))
OLfilenames = glob(os.path.join(basedir, '**', satellite + '_OL_1_ERR____**.zip'), recursive=True)
OLfilename = None
for OLf in OLfilenames:
    OLtimes = os.path.split(OLf)[-1][16:47].split('_')
    OLt0 = datetime(int(OLtimes[0][:4]), int(OLtimes[0][4:6]), int(OLtimes[0][6:8]), int(OLtimes[0][9:11]), int(OLtimes[0][11:13]), int(OLtimes[0][13:15]))
    OLt1 = datetime(int(OLtimes[1][:4]), int(OLtimes[1][4:6]), int(OLtimes[1][6:8]), int(OLtimes[1][9:11]), int(OLtimes[1][11:13]), int(OLtimes[1][13:15]))
    if OLt0 <= SYt0 and OLt1 >= SYt1:
        OLfilename = OLf
        break
SLfilenames = glob(os.path.join(basedir, '**', satellite + '_SL_1_RBT____**.zip'), recursive=True)
SLfilename = None
for SLf in SLfilenames:
    SLtimes = os.path.split(SLf)[-1][16:47].split('_')
    SLt0 = datetime(int(SLtimes[0][:4]), int(SLtimes[0][4:6]), int(SLtimes[0][6:8]), int(SLtimes[0][9:11]), int(SLtimes[0][11:13]), int(SLtimes[0][13:15]))
    SLt1 = datetime(int(SLtimes[1][:4]), int(SLtimes[1][4:6]), int(SLtimes[1][6:8]), int(SLtimes[1][9:11]), int(SLtimes[1][11:13]), int(SLtimes[1][13:15]))
    if SLt0 <= SYt0 and SLt1 >= SYt1:
        SLfilename = SLf
        break

if not os.path.isfile(OLfilename):
    print('Not found OL_1_ERR file: {}'.format(OLfilename))
    sys.exit(1)

if not os.path.isfile(SLfilename):
    print('Not found SL_1_RBT file: {}'.format(SLfilename))
    sys.exit(1)

if not os.path.isfile(SYfilename):
    print('Not found SY_2_SYN file: {}'.format(SYfilename))
    sys.exit(1)


# POPCORN neural network model
class POPCORNDNN_2layers(pl.LightningModule):
    def __init__(self, Ninputs, N1sthidden, N2ndhidden, Noutputs, lr=5e-5):
        super().__init__()
        self.lr = lr
        self.nn = nn.Sequential(
            nn.Linear(Ninputs, N1sthidden),
            nn.ReLU(),
            nn.Linear(N1sthidden, N2ndhidden),
            nn.ReLU(),
            nn.Linear(N2ndhidden, Noutputs)
        )

    def forward(self, x):
        return self.nn(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        x_hat = self.nn(x)
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss, on_epoch=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        x_hat = self.nn(x)
        loss = F.mse_loss(x_hat, y)
        self.log('val_loss', loss, on_epoch=True, logger=True)
        return loss


imputer_scalerCORR = joblib.load('models/FINAL_CORR_imputer_scaler.joblib')
CORRlayers = [156, 128]
modelCORR = POPCORNDNN_2layers(Ninputs=156, N1sthidden=CORRlayers[0], N2ndhidden=CORRlayers[1], Noutputs=5, lr=1e-4)
modelCORR.load_state_dict(torch.load('models/CORR_FINAL.pt'))
modelCORR.eval()

S3data = loadS3_SY_OL_SL(SYfilename, OLfilename, SLfilename)

if S3data is None:
    print('ERROR IN LOADING DATA', SYfilename)
    sys.exit(0)

GEOMETRYvariables = ['altitude', 'SYN_O_VAA', 'SYN_O_VZA', 'SYN_O_SAA', 'SYN_O_SZA', 'SYN_SN_VAA', 'SYN_SN_VZA', 'SYN_SO_VAA', 'SYN_SO_VZA', 'SYN_O_scattering_angle', 'SYN_SO_scattering_angle', 'SYN_SN_scattering_angle']
SATELLITOBSERVATIONvariables = ['SL1_S1_reflectance_nadir', 'SL1_S1_reflectance_oblique', 'SL1_S2_reflectance_nadir', 'SL1_S2_reflectance_oblique', 'SL1_S3_reflectance_nadir', 'SL1_S3_reflectance_oblique', 'SL1_S4_reflectance_nadir', 'SL1_S4_reflectance_oblique', 'SL1_S5_reflectance_nadir', 'SL1_S5_reflectance_oblique', 'SL1_S6_reflectance_nadir', 'SL1_S6_reflectance_oblique', 'OL1_Oa01_reflectance', 'OL1_Oa02_reflectance', 'OL1_Oa03_reflectance', 'OL1_Oa04_reflectance', 'OL1_Oa05_reflectance', 'OL1_Oa06_reflectance', 'OL1_Oa07_reflectance', 'OL1_Oa08_reflectance', 'OL1_Oa09_reflectance', 'OL1_Oa10_reflectance', 'OL1_Oa11_reflectance', 'OL1_Oa12_reflectance', 'OL1_Oa13_reflectance', 'OL1_Oa14_reflectance', 'OL1_Oa15_reflectance', 'OL1_Oa16_reflectance', 'OL1_Oa17_reflectance', 'OL1_Oa18_reflectance', 'OL1_Oa19_reflectance', 'OL1_Oa20_reflectance', 'OL1_Oa21_reflectance']
SYNLEVEL2variables = ['SYN_AOD550', 'SYN_AOD550err', 'SYN_AE550', 'SYN_AMIN', 'SYN_SYN_no_slo', 'SYN_SYN_no_sln', 'SYN_SYN_no_olc', 'SYN_SDR_Oa01', 'SYN_SDR_Oa02', 'SYN_SDR_Oa03', 'SYN_SDR_Oa04', 'SYN_SDR_Oa05', 'SYN_SDR_Oa06', 'SYN_SDR_Oa07', 'SYN_SDR_Oa08', 'SYN_SDR_Oa09', 'SYN_SDR_Oa10', 'SYN_SDR_Oa11', 'SYN_SDR_Oa12', 'SYN_SDR_Oa16', 'SYN_SDR_Oa17', 'SYN_SDR_Oa18', 'SYN_SDR_Oa21', 'SYN_SDR_S1N', 'SYN_SDR_S1O', 'SYN_SDR_S2N', 'SYN_SDR_S2O', 'SYN_SDR_S3N', 'SYN_SDR_S3O', 'SYN_SDR_S5N', 'SYN_SDR_S5O', 'SYN_SDR_S6N', 'SYN_SDR_S6O']

dataMASK = np.logical_and.reduce((
    S3data['SYN_SYN_land'],
    S3data['SYN_SLN_day'],
    ~S3data['SYN_SLN_twilight'],
    ~S3data['SYN_SYN_too_low'],
    ~S3data['SYN_SYN_high_error'],
    ~S3data['SYN_SYN_AOT_climato'],
    ~S3data['SYN_CLOUD'],
    ~S3data['SYN_CLOUD_AMBIGUOUS'],
    ~S3data['SYN_SNOW_ICE'],
    ~S3data['SYN_SLO_snow'],
    ~S3data['SYN_SLN_snow'],
    ~S3data['SYN_SYN_snow_risk'],
    ~S3data['OL1_flags_cosmetic'],
    ~S3data['SYN_OLC_cosmetic'],
    ~S3data['SYN_SLN_cosmetic'],
))

bestdataMASK = np.logical_and.reduce((
    S3data['SYN_SYN_land'],
    ~S3data['SYN_SYN_AOT_climato'],
    ~S3data['SYN_SYN_aerosol_filled'],
    ~S3data['SYN_CLOUD'],
    ~S3data['SYN_CLOUD_AMBIGUOUS'],
    ~S3data['SYN_SYN_partly_cloudy'],
    S3data['SYN_SLN_day'],
    ~S3data['SYN_SLN_twilight'],
    ~S3data['SYN_SYN_too_low'],
    ~S3data['SYN_SYN_high_error'],
    ~S3data['SYN_CLOUD_MARGIN'],
    ~S3data['SYN_SNOW_ICE'],
    ~S3data['SYN_SLO_snow'],
    ~S3data['SYN_SLN_snow'],
    ~S3data['SYN_SYN_snow_risk'],
    ~S3data['OL1_flags_cosmetic'],
    ~S3data['SYN_OLC_cosmetic'],
    ~S3data['SYN_SLN_cosmetic'],
))

os.makedirs(OUTPUTDIR, exist_ok=True)

CORRinputvariables = GEOMETRYvariables + SATELLITOBSERVATIONvariables + SYNLEVEL2variables
CORRoutputvariables = ['AOD550_approximationerror', 'AOD440_approximationerror', 'AOD500_approximationerror', 'AOD675_approximationerror', 'AOD870_approximationerror']

inputsminmaxCORR = imputer_scalerCORR['inputsminmax']
scaler_inputsCORR = imputer_scalerCORR['scaler_inputs']
scaler_outputsCORR = imputer_scalerCORR['scaler_outputs']
imputerCORR = imputer_scalerCORR['imputer']

S3dataMASK = np.logical_and.reduce((~S3data['SYN_AOD550'].mask, ~S3data['SYN_AE550'].mask, dataMASK))
bestS3dataMASK = np.logical_and.reduce((~S3data['SYN_AOD550'].mask, ~S3data['SYN_AE550'].mask, bestdataMASK))

# save data dict
saves = {
    'AOD_CORR': np.nan * np.zeros((5, S3dataMASK.shape[0], S3dataMASK.shape[1])),
    'bestdataMASK': bestdataMASK
}

# Run post-process correction
Npixels = S3dataMASK.sum()
coordsii, coordsjj = np.where(S3dataMASK)
MAXpixelsatonce = 250000
for indx in range(0, Npixels, MAXpixelsatonce):
    this_coordsii = coordsii[indx:min(indx + MAXpixelsatonce, Npixels)]
    this_coordsjj = coordsjj[indx:min(indx + MAXpixelsatonce, Npixels)]
    N = len(this_coordsii)
    if N == 0:
        continue
    inputs = np.nan * np.ones((N, len(CORRinputvariables)))
    for jj, ipvar in enumerate(CORRinputvariables):
        inputs[:, jj] = np.clip(S3data[ipvar][this_coordsii, this_coordsjj], a_min=inputsminmaxCORR[0, jj], a_max=inputsminmaxCORR[1, jj])
    inputs = np.hstack((inputs, np.isnan(inputs).astype(float)))
    predicted_approx_err = scaler_outputsCORR.inverse_transform(modelCORR.forward(torch.Tensor(scaler_inputsCORR.transform(imputerCORR.transform(inputs)))).detach().numpy())
    saves['AOD_CORR'][0, this_coordsii, this_coordsjj] = np.clip(S3data['SYN_AOD550'][this_coordsii, this_coordsjj] * (550.0 / 550.0)**-S3data['SYN_AE550'][this_coordsii, this_coordsjj] + predicted_approx_err[:, 0], a_min=0.005, a_max=10)
    saves['AOD_CORR'][1, this_coordsii, this_coordsjj] = np.clip(S3data['SYN_AOD550'][this_coordsii, this_coordsjj] * (440.0 / 550.0)**-S3data['SYN_AE550'][this_coordsii, this_coordsjj] + predicted_approx_err[:, 1], a_min=0.005, a_max=10)
    saves['AOD_CORR'][2, this_coordsii, this_coordsjj] = np.clip(S3data['SYN_AOD550'][this_coordsii, this_coordsjj] * (500.0 / 550.0)**-S3data['SYN_AE550'][this_coordsii, this_coordsjj] + predicted_approx_err[:, 2], a_min=0.005, a_max=10)
    saves['AOD_CORR'][3, this_coordsii, this_coordsjj] = np.clip(S3data['SYN_AOD550'][this_coordsii, this_coordsjj] * (675.0 / 550.0)**-S3data['SYN_AE550'][this_coordsii, this_coordsjj] + predicted_approx_err[:, 3], a_min=0.005, a_max=10)
    saves['AOD_CORR'][4, this_coordsii, this_coordsjj] = np.clip(S3data['SYN_AOD550'][this_coordsii, this_coordsjj] * (870.0 / 550.0)**-S3data['SYN_AE550'][this_coordsii, this_coordsjj] + predicted_approx_err[:, 4], a_min=0.005, a_max=10)

# write outputs to a netCDF file
outputfile = os.path.join(OUTPUTDIR, 'POPCORN_CORR_{}.nc'.format(SYfilename_original.replace('.zip', '')))
ncout = Dataset(outputfile, 'w', format='NETCDF4')
# Add attributes
ncout.History = 'File generated on {} (UTC) by {}'.format(datetime.utcnow().strftime('%c'), os.path.basename(__file__))
ncout.Description = 'POPCORN post-process corrected Sentinel-3 SYNERGY AOD'
ncout.AlgorithmDevelopers = 'Finnish Meteorological Institute and University of Eastern Finland'
ncout.AlgorithmContactInformation = 'Antti Lipponen (antti.lipponen@fmi.fi)'
ncout.SYNyear = '{:d}'.format(SYt0.year)
ncout.SYNmonth = '{:d}'.format(SYt0.month)
ncout.SYNday = '{:d}'.format(SYt0.day)
ncout.SYNhour = '{:d}'.format(SYt0.hour)
ncout.SYNminute = '{:d}'.format(SYt0.minute)
ncout.originalSYNfile = SYfilename_original
ncout.start_time_str = S3data['start_time_str']
ncout.stop_time_str = S3data['stop_time_str']

# create dimensions
ncout.createDimension('Nwavelengths', 5)
ncout.createDimension('rows', saves['AOD_CORR'].shape[1])
ncout.createDimension('columns', saves['AOD_CORR'].shape[2])

# save coordinates
ncout_latitude = ncout.createVariable('Latitude', 'f4', ('rows', 'columns'), zlib=True, complevel=4, least_significant_digit=6)
ncout_latitude[:] = S3data['lat']
ncout_latitude.standard_name = 'latitude'
ncout_latitude.units = "degrees_north"
ncout_longitude = ncout.createVariable('Longitude', 'f4', ('rows', 'columns'), zlib=True, complevel=4, least_significant_digit=6)
ncout_longitude[:] = S3data['lon']
ncout_longitude.standard_name = 'longitude'
ncout_longitude.units = "degrees_east"
ncout_wavelengths = ncout.createVariable('Wavelength', 'f4', ('Nwavelengths', ))
ncout_wavelengths[:] = [440.0, 500.0, 550.0, 675.0, 870.0]
ncout_wavelengths.standard_name = 'wavelength'
ncout_wavelengths.units = "nm"

# save data
ncout_dMASK = ncout.createVariable('bestDataMask', 'u1', ('rows', 'columns'), zlib=True, complevel=4)
ncout_dMASK.standard_name = 'Mask indicating the best quality data'
ncout_dMASK.coordinates = 'Wavelength Latitude Longitude'
ncout_dMASK.units = "0/1"
ncout_dMASK[:] = saves['bestdataMASK'].astype(np.uint8)
ncout_dAOD = ncout.createVariable('AOD', 'f4', ('Nwavelengths', 'rows', 'columns'), zlib=True, complevel=4, least_significant_digit=2)
ncout_dAOD.standard_name = 'Aerosol Optical Depth'
ncout_dAOD.coordinates = 'Wavelength Latitude Longitude'
ncout_dAOD.units = ""
ncout_dAOD[:] = saves['AOD_CORR']

# close nc file
ncout.close()

print('Processing done!')
print('Data saved to {}'.format(outputfile))
print('Thank you for using POPCORN!')
print('')
