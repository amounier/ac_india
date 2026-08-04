#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:59:08 2026

@author: amounier
"""

import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import xarray as xr 
import cartopy.crs as ccrs
import matplotlib
from shapely.geometry import mapping
import tqdm
from rioxarray.exceptions import NoDataInBounds
import pickle
from shapely.ops import unary_union

from utils import (save_figure,
                   draw_state_map,
                   dict_name_code,
                   blank_map,
                   dict_name_geom,
                   get_ssp_palette,
                   get_ssp_name,)

EXPERIMENTS_DICT = {'ssp1_2_6':['cmcc_esm2','iitm_esm','hadgem3_gc31_mm','miroc6','cnrm_cm6_1'],
                    'ssp2_4_5':['cmcc_esm2','iitm_esm','hadgem3_gc31_ll','miroc6','cnrm_cm6_1'],
                    'ssp3_7_0':['cmcc_cm2_sr5','iitm_esm','canesm5','miroc6','cnrm_cm6_1'],
                    'ssp5_8_5':['cmcc_esm2','iitm_esm','hadgem3_gc31_mm','canesm5','miroc6','cnrm_cm6_1'],}


def get_cdd_data(threshold=26, source='iea', columns_format_name=True):
    if source == 'iea_india':
        file = os.path.join('data','IEA_CMCC',f"IEA_CMCC_CDD{threshold}monthlyworldbypopallmonths.csv")
        data = pd.read_csv(file,skiprows=9,encoding='latin-1')
        data = data[data['ISO3']=='IND']
        # format state name
        data = data[['Date',f'CDD{threshold}']]
        data['Date'] = pd.to_datetime(data.Date, format='%Y-%m-%d')
        data = data.set_index('Date')
        data = data.groupby(data.index.year).sum()
    
    if source == 'iea':
        file = os.path.join('data','IEA_CMCC',f"IEA_CMCC_CDD{threshold}monthlysubnationalbypopallmonths.csv")
        data = pd.read_csv(file,skiprows=9,encoding='latin-1')
        data = data[data['COUNTRY ISO3']=='IND']
        # format state name
        data['Territory'] = data['Territory'].str.replace('Orissa','Odisha').replace('Uttaranchal','Uttarakhand')
        data = data[['Territory','Date',f'CDD{threshold}']]
        data['Date'] = pd.to_datetime(data.Date, format='%Y-%m-%d')
        data = data.set_index(['Date','Territory'])
        data = data.unstack()
        data.columns = data.columns.droplevel(0)
        data = data.groupby(data.index.year).sum()
        
    if source == 'own':
        file = os.path.join('data','CDD-States-2011-2023-popweighted.xlsx')
        data = pd.read_excel(file,sheet_name=f'CDD{threshold}')
        # format state name
        data['States'] = data['States'].str.replace('&','and')
        data = data.set_index('States').transpose()
    
    if not columns_format_name:
        data = data.rename(columns=dict_name_code)
    return data


def add_cdd_data(base_year=2024, mean_period=10, threshold=26, source='iea', 
                 columns_format_name=True,base_data=None):
    data = get_cdd_data(threshold=threshold, source='iea',columns_format_name=columns_format_name)
    data = data[data.index.isin(list(range(base_year-mean_period+1,base_year+1)))]
    data = data.mean()
    data = pd.DataFrame(data).rename(columns={0:f'cdd{threshold}'})
    
    data = data.rename(columns={k:f'{k}_{base_year}' for k in data.columns})
    
    if base_data is not None:
        base_data = base_data.join(data)
    else:
        base_data = data
    return base_data


def get_population():
    data = os.path.join('data','gpw-v4-population-count-rev11_totpop_15_min_nc','gpw_v4_population_count_rev11_15_min.nc')
    gridded_pop = xr.open_dataset(data)
    gridded_pop = gridded_pop[list(gridded_pop.var())[0]]
    gridded_pop = gridded_pop.sel(raster=5)
    gridded_pop = gridded_pop.sel(latitude=slice(38.,6.), longitude=slice(68., 98.))
    return gridded_pop


def get_temperature_data(model,scenario,period):
    folder = os.path.join('data','CMIP6',f'tas_{scenario}_{model}_{period}')
    file = [f for f in os.listdir(folder) if '.nc' in f][0]
    ds = xr.open_dataset(os.path.join(folder,file))
    da = ds.tas - 273.15
    return da

def get_state_temperature_data(state,model,scenario,period):
    da = get_temperature_data(model, scenario, period)
    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    da.rio.write_crs("epsg:4326", inplace=True)
    
    states_geometry = dict_name_geom
    state_geometry = states_geometry.get(state)
    
    flag = False
    try:
        da_clipped = da.copy().rio.clip([state_geometry])
            
    except NoDataInBounds:
        da_clipped = da
        flag = True
        
    if len(da_clipped.lat.values) == 1 or len(da_clipped.lon.values) == 1 or flag:
        longitude, latitude = tuple(state_geometry.centroid.coords)[0]
        match_point = da.sel(lat=latitude,lon=longitude,method='nearest')
        da_clipped = match_point.copy()
        
    return da_clipped


def get_state_temperature_data_population_weighted(state,model,scenario):
    data = None
    
    for period in ['2015_2049','2050_2100']:
        states_geometry = dict_name_geom
        state_geometry = states_geometry.get(state)
        
        da = get_state_temperature_data(state, model, scenario, period)
        
        if len(da.shape) == 3:
            pop = get_population()
            pop.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            pop.rio.write_crs("epsg:4326", inplace=True)
            
            pop_clipped = pop.copy().rio.clip([state_geometry])
                
            pop_resample = pop_clipped.interp(longitude=da.lon,latitude=da.lat, method='nearest')
            pop_resample = pop_resample/pop_resample.sum()
            
            da_pop_weighted = da * pop_resample
            da_pop_weighted_mean = da_pop_weighted.sum(dim=['lat','lon'],skipna=True)
            
        elif len(da.shape) == 1:
            da_pop_weighted_mean = da
        
        data_dict = dict()
        data_dict['date'] = da_pop_weighted_mean.time.values
        data_dict['temperature'] = da_pop_weighted_mean.values
        if data is None:
            data = pd.DataFrame().from_dict(data_dict)
        else:
            data = pd.concat([data, pd.DataFrame().from_dict(data_dict)])
    data['date'] = [e.strftime('%Y-%m-%d') for e in data.date]
    data['date'] = pd.to_datetime(data.date,errors='coerce')
    data = data.dropna()
    data = data.set_index('date')
    data['experiment'] = [scenario]*len(data)
    data['model'] = [model]*len(data)
    data['state'] = [state]*len(data)
    return data


def format_state_temperature_data_population_weighted(force=False):
    data = None
    save_folder = os.path.join('data')
    save_name = 'temperature.parquet'
    
    if save_name not in os.listdir(save_folder) or force:
        for state in tqdm.tqdm(list(dict_name_geom.keys())):
            for experiment,models_list in EXPERIMENTS_DICT.items():
                for model in models_list:
                    if data is None:
                        data = get_state_temperature_data_population_weighted(state,model,experiment)
                    else:
                        data = pd.concat([data,get_state_temperature_data_population_weighted(state,model,experiment)])
            
        data.to_parquet(os.path.join(save_folder,save_name))
        
    data = pd.read_parquet(os.path.join(save_folder,save_name))
    return data


def format_cdd(threshold,force=False):
    temperature = format_state_temperature_data_population_weighted(force=force)
    temperature = temperature.reset_index()
    temperature['year'] = temperature.date.dt.year
    temperature[f'cdd{threshold}'] = (temperature.temperature - threshold).clip(lower=0.)
    
    cdd = temperature[['year','state','experiment','model',f'cdd{threshold}']].groupby(['year','state','experiment','model']).sum()
    return cdd


def export_cdd(threshold,force=False):
    save_folder = os.path.join('data')
    iea_cdd_calibration_file = 'cdd_calibration.csv'
    cdd_projection_file = 'cdd_projection.csv'
    
    cdd = format_cdd(threshold=threshold,force=force)
    
    iea_data = get_cdd_data(threshold=threshold, source='iea')
    iea_data = iea_data[iea_data.index.isin(list(range(2011,2025)))]
    iea_mean = iea_data[iea_data.index.isin(list(range(2015,2025)))].mean().to_dict()
    
    cdd_projection = None
    iea_cdd_calibration = {'state':[],'experiment':[],'model':[],f'cdd{threshold}_calibration':[]}
    iea_missing_states = set()
    
    if cdd_projection_file not in os.listdir(save_folder) or force:
        for scenario in EXPERIMENTS_DICT.keys():
            for state in dict_name_geom.keys():
                if state not in iea_data.columns:
                    iea_missing_states.add(state)
                    continue 
                
                cdd_state = cdd[cdd.index.get_level_values('state')==state]
                cdd_state = cdd_state[cdd_state.index.get_level_values('experiment')==scenario]
                cdd_state = cdd_state.reset_index()
                
                # recalibration
                iea_mean_state = iea_mean.get(state)
                cdd_state_mean = cdd_state[cdd_state.year.isin(list(range(2015,2025)))]
                cdd_state_mean = cdd_state_mean[['model',f'cdd{threshold}']].groupby('model').mean()
                cdd_state_mean = cdd_state_mean[f'cdd{threshold}'].to_dict()
                
                for model,mean_val in cdd_state_mean.items():
                    iea_cdd_calibration['state'].append(state)
                    iea_cdd_calibration['experiment'].append(scenario)
                    iea_cdd_calibration['model'].append(model)
                    iea_cdd_calibration[f'cdd{threshold}_calibration'].append(iea_mean_state-mean_val)
                    
                cdd_update = []
                for model in cdd_state.model:
                    cdd_update.append(iea_mean_state-cdd_state_mean.get(model))
                    
                cdd_update = np.asarray(cdd_update)
                cdd_state[f'cdd{threshold}'] = cdd_state[f'cdd{threshold}'] + cdd_update
                cdd_state[f'cdd{threshold}'] = cdd_state[f'cdd{threshold}'].clip(lower=0.)
                
                if cdd_projection is None:
                    cdd_projection = cdd_state
                else:
                    cdd_projection = pd.concat([cdd_projection,cdd_state])
        
        iea_cdd_calibration = pd.DataFrame().from_dict(iea_cdd_calibration)
        iea_cdd_calibration.to_csv(os.path.join(save_folder,iea_cdd_calibration_file),index=False)
        cdd_projection.to_csv(os.path.join(save_folder,cdd_projection_file),index=False)
        with open(os.path.join(save_folder,'iea_missing_states.pickle'), 'wb') as handle:
            pickle.dump(iea_missing_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    else:
        cdd_projection = pd.read_csv(os.path.join(save_folder,cdd_projection_file))
        iea_cdd_calibration = pd.read_csv(os.path.join(save_folder,iea_cdd_calibration_file))
        with open(os.path.join(save_folder,'iea_missing_states.pickle'), 'rb') as handle:
            iea_missing_states = pickle.load(handle)
    
    return cdd_projection, iea_cdd_calibration, iea_missing_states


def get_india_temperature_data(model,scenario,period):
    da = get_temperature_data(model, scenario, period)
    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    da.rio.write_crs("epsg:4326", inplace=True)
    
    india_shape = unary_union(list(dict_name_geom.values()))
    da_clipped = da.copy().rio.clip([india_shape])
    
    return da_clipped


def get_india_temperature_data_population_weighted(model,scenario):
    data = None
    
    for period in ['2015_2049','2050_2100']:
        da = get_india_temperature_data(model, scenario, period)
        
        pop = get_population()
        pop.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
        pop.rio.write_crs("epsg:4326", inplace=True)
        
        india_shape = unary_union(list(dict_name_geom.values()))
        pop_clipped = pop.copy().rio.clip([india_shape])
            
        pop_resample = pop_clipped.interp(longitude=da.lon,latitude=da.lat, method='nearest')
        pop_resample = pop_resample/pop_resample.sum()
        
        da_pop_weighted = da * pop_resample
        da_pop_weighted_mean = da_pop_weighted.sum(dim=['lat','lon'],skipna=True)
            

        data_dict = dict()
        data_dict['date'] = da_pop_weighted_mean.time.values
        data_dict['temperature'] = da_pop_weighted_mean.values
        if data is None:
            data = pd.DataFrame().from_dict(data_dict)
        else:
            data = pd.concat([data, pd.DataFrame().from_dict(data_dict)])
    data['date'] = [e.strftime('%Y-%m-%d') for e in data.date]
    data['date'] = pd.to_datetime(data.date,errors='coerce')
    data = data.dropna()
    data = data.set_index('date')
    data['experiment'] = [scenario]*len(data)
    data['model'] = [model]*len(data)
    return data


def format_india_temperature_data_population_weighted(force=False):
    data = None
    save_folder = os.path.join('data')
    save_name = 'temperature_INDIA.parquet'
    
    if save_name not in os.listdir(save_folder) or force:
        for experiment,models_list in EXPERIMENTS_DICT.items():
            for model in models_list:
                if data is None:
                    data = get_india_temperature_data_population_weighted(model,experiment)
                else:
                    data = pd.concat([data,get_india_temperature_data_population_weighted(model,experiment)])
            
        data.to_parquet(os.path.join(save_folder,save_name))
        
    data = pd.read_parquet(os.path.join(save_folder,save_name))
    return data


def format_cdd_INDIA(threshold,force=False):
    temperature = format_india_temperature_data_population_weighted(force=force)
    temperature = temperature.reset_index()
    temperature['year'] = temperature.date.dt.year
    temperature[f'cdd{threshold}'] = (temperature.temperature - threshold).clip(lower=0.)
    
    cdd = temperature[['year','experiment','model',f'cdd{threshold}']].groupby(['year','experiment','model']).sum()
    return cdd

def export_cdd_INDIA(threshold,force=False):
    save_folder = os.path.join('data')
    iea_cdd_calibration_file = 'cdd_calibration_INDIA.csv'
    cdd_projection_file = 'cdd_projection_INDIA.csv'
    
    cdd = format_cdd_INDIA(threshold=threshold,force=force)
    
    iea_data = get_cdd_data(threshold=threshold, source='iea_india')
    iea_data = iea_data[iea_data.index.isin(list(range(2011,2025)))]
    iea_mean = iea_data[iea_data.index.isin(list(range(2015,2025)))].mean().to_dict()
    
    cdd_projection = None
    iea_cdd_calibration = {'experiment':[],'model':[],f'cdd{threshold}_calibration':[]}
    
    if cdd_projection_file not in os.listdir(save_folder) or force:
        for scenario in EXPERIMENTS_DICT.keys():
                
            cdd_state = cdd[cdd.index.get_level_values('experiment')==scenario]
            cdd_state = cdd_state.reset_index()
            
            # recalibration
            iea_mean_state = iea_mean.get(f'CDD{threshold}')
            cdd_state_mean = cdd_state[cdd_state.year.isin(list(range(2015,2025)))]
            cdd_state_mean = cdd_state_mean[['model',f'cdd{threshold}']].groupby('model').mean()
            cdd_state_mean = cdd_state_mean[f'cdd{threshold}'].to_dict()
            
            for model,mean_val in cdd_state_mean.items():
                iea_cdd_calibration['experiment'].append(scenario)
                iea_cdd_calibration['model'].append(model)
                iea_cdd_calibration[f'cdd{threshold}_calibration'].append(iea_mean_state-mean_val)
                
            cdd_update = []
            for model in cdd_state.model:
                cdd_update.append(iea_mean_state-cdd_state_mean.get(model))
                
            cdd_update = np.asarray(cdd_update)
            cdd_state[f'cdd{threshold}'] = cdd_state[f'cdd{threshold}'] + cdd_update
            cdd_state[f'cdd{threshold}'] = cdd_state[f'cdd{threshold}'].clip(lower=0.)
            
            if cdd_projection is None:
                cdd_projection = cdd_state
            else:
                cdd_projection = pd.concat([cdd_projection,cdd_state])
        
        iea_cdd_calibration = pd.DataFrame().from_dict(iea_cdd_calibration)
        iea_cdd_calibration.to_csv(os.path.join(save_folder,iea_cdd_calibration_file),index=False)
        cdd_projection.to_csv(os.path.join(save_folder,cdd_projection_file),index=False)
        
    else:
        cdd_projection = pd.read_csv(os.path.join(save_folder,cdd_projection_file))
        iea_cdd_calibration = pd.read_csv(os.path.join(save_folder,iea_cdd_calibration_file))
    
    return cdd_projection, iea_cdd_calibration

#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    #%% Comparaison entre les données IEA x CMCC et les calculs perso
    if False:
        threshold = 26 #°C
        iea_data = get_cdd_data(threshold=threshold, source='iea')
        iea_data = iea_data[iea_data.index.isin(list(range(2011,2025)))]
        
        data = get_cdd_data(threshold=threshold, source='own')
        
        for state in dict_name_code.keys():
            if state not in data.columns or state not in iea_data.columns:
                continue
            
            state_code = dict_name_code.get(state)
            
            fig,ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.plot(iea_data[state],label='IEA $\\times$ CMCC')
            ax.plot(data[state],label='Own computation')
            ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_ylabel(f'CDD{threshold} (°C.days)')
            ax.set_title(state)
            save_path = os.path.join(f'cdd{threshold}',f'cdd{threshold}_state{state_code}')
            save_figure(save_path)
            plt.show()
            plt.close()
            
    #%% Carte des CDD
    if False:
        mean_period = 10 #yrs
        year = 2021
        
        for threshold in [18,21,23,26]: #°C
            data = add_cdd_data(base_year=year,mean_period=mean_period,threshold=threshold,source='iea')
            data = data.to_dict().get(f'cdd{threshold}_{year}')
            
            draw_state_map(data,cbar_min=0,cbar_max=np.quantile(np.asarray(list(data.values())),0.9999),
                           cbar_label=f'Population-ponderated CDD{threshold} (°C.days)',cmap=plt.get_cmap('YlOrRd'),
                           map_title = f'{year-mean_period+1} $-$ {year}',
                           save=os.path.join('maps',f'cdd{threshold}_{year}_period{mean_period}'))
    
    #%% Téléchargement des données CDS
    if False:
        import cdsapi
        
        for experiment,models_list in EXPERIMENTS_DICT.items():
            for model in models_list:
                print('-'*40)
                print(f'Scenario : {experiment}, Model : {model}')
                print('-'*40,'\n')
                
                # years = [str(e) for e in list(range(2015,2050))]
                years = [str(e) for e in list(range(2050,2101))]
                
                dataset = "projections-cmip6"
                request = {"temporal_resolution": "daily",
                           "experiment": experiment,
                           "variable": "near_surface_air_temperature",
                           "model": model,
                           "year": years,
                           "month": ["01", "02", "03",
                                     "04", "05", "06",
                                     "07", "08", "09",
                                     "10", "11", "12"],
                           "area": [38, 68, 6, 98]
                           }
                
                save_path = os.path.join('data','CMIP6')
                save_file = f'tas_{experiment}_{model}_{years[0]}_{years[-1]}'
                
                if save_file in os.listdir(save_path):
                    print('Already downloaded.\n')
                    continue 
                
                client = cdsapi.Client()
                client.retrieve(dataset, request).download()
                
                downloaded_file = [e for e in os.listdir() if'.zip' in e][0]
                
                subprocess.run(f"unzip {downloaded_file} -d {save_file}",shell=True)
                subprocess.run(f"mv {save_file} {os.path.join(save_path,save_file)}",shell=True)
                subprocess.run(f"rm {downloaded_file}",shell=True)
                
                
    #%% Données de population
    if True: 
        cmap = plt.get_cmap('viridis')
        
        india_shape = unary_union(list(dict_name_geom.values()))
        
        pop = get_population()
        pop.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
        pop.rio.write_crs("epsg:4326", inplace=True)
        pop_clipped = pop.copy().rio.clip([india_shape])
        
        fig,ax = blank_map()
        pop_clipped.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False, norm=matplotlib.colors.LogNorm(),vmin=1e4)
        
        cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
        norm = matplotlib.colors.LogNorm(vmin=1e4, vmax=pop_clipped.max())
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        cbar_label_var = 'Population in 2020 (inhab)'
        _ = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend='min', extendfrac=0.02)
        
        ax.set_title('')
        plt.show()
            
    #%% Mise en forme des données et calcul des CDD, par état indien
    
    # premier test, carte des températures
    if False:
        
        for state in dict_name_geom.keys():
            if state != 'Punjab':
                continue
            print(state)
            model = 'miroc6'
            scenario = 'ssp2_4_5'
            period = '2050_2100'
            
            # state = 'Assam'
            
            time_step = '2051-05-11'
            xr_time = xr.DataArray([1], coords={"time": pd.to_datetime([time_step])}, dims=["time"])
            
            da = get_state_temperature_data(state, model, scenario, period)
            
            if len(da.shape) == 1:
                continue 
            
            da = da.sel(time=xr_time.time.values[0],method='nearest')
            
            cmap = 'YlOrRd'
            var_label = 'Mean daily temperature (°C)'
            
            fig,ax = blank_map()
            da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
            
            cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
            posn = ax.get_position()
            cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
            norm = matplotlib.colors.Normalize(vmin=da.min(), vmax=da.max())
            mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            
            cbar_label_var = var_label
            _ = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend='neither', extendfrac=0.02)
            
            ax.set_title(state + ' $-$ ' + time_step)
            plt.show()
        
    
    # formatage des températures et calcul des CDD26
    if False:
        threshold = 26 #°C
        # scenario = 'ssp2_4_5'
        scenario = 'ssp5_8_5'
        
        iea_data = get_cdd_data(threshold=threshold, source='iea')
        iea_data = iea_data[iea_data.index.isin(list(range(2011,2025)))]
        
        cdd_projection,_,missing_states = export_cdd(threshold)
        
        for state in dict_name_geom.keys():
            if state in missing_states:
                continue 
            
            cdd_state = cdd_projection[cdd_projection.state==state]
            
        
            # évolution de chaque modèle
            if False:
                cdd_state = cdd_state[cdd_state.experiment==scenario]
                
                ls_list = ['solid','dotted','dashed','dashdot',]
                cmap = plt.get_cmap('viridis')
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                ax.plot(iea_data[state],color='k',label='IEA $\\times$ CMCC')
                
                models_list = EXPERIMENTS_DICT.get(scenario)
                for idx,mod in enumerate(models_list):
                    cdd_state_mod = cdd_state[cdd_state.model==mod][['year',f'cdd{threshold}']].set_index('year')
                    ax.plot(cdd_state_mod,color=cmap(1-(idx+1)/(len(models_list))),label=mod,ls=ls_list[idx%len(ls_list)])
                
                ax.set_title(f'{state} - {scenario}')
                ax.set_ylim(bottom=0.)
                ax.legend()
                ax.set_ylabel(f'CDD{threshold} (°C.days)')
                plt.show()
            
            # moyenne des modèles
            if True:
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                ax.plot(iea_data[state],color='k',label='IEA $\\times$ CMCC',zorder=5)
                
                for scenario in EXPERIMENTS_DICT.keys():
                    cdd_state_scenario = cdd_state[cdd_state.experiment==scenario]
                    
                    cdd_state_mean_mod = cdd_state_scenario[['year',f'cdd{threshold}']].groupby('year').mean()
                    cdd_state_std_mod = cdd_state_scenario[['year',f'cdd{threshold}']].groupby('year').std()
                
                
                    ax.plot(cdd_state_mean_mod,color=get_ssp_palette(scenario),label=get_ssp_name(scenario))
                    
                    if scenario in ['ssp1_2_6', 'ssp5_8_5']:
                        ax.fill_between(cdd_state_mean_mod.index,
                                        cdd_state_mean_mod.cdd26+cdd_state_std_mod.cdd26,
                                        cdd_state_mean_mod.cdd26-cdd_state_std_mod.cdd26,
                                        alpha=0.2,color=get_ssp_palette(scenario))
                    
                ax.set_title(f'{state}')
                ax.set_ylim(bottom=0.)
                ax.set_xlim(left=2015)
                ax.legend()
                ax.set_ylabel(f'CDD{threshold} (°C.days)')
                plt.show()
    
    
    #%% cartes et graphes à l'échelle nationale pour illustration
    if True:
        
        # cartes de températures (difference entre 2015-2035 et 2080-2100, SSP3-7.0)
        if True:
            experiment = 'ssp3_7_0'
            period_ini = [2015,2035]
            period_end = [2080,2100]
            
            india_shape = unary_union(list(dict_name_geom.values()))
            
            for mod in EXPERIMENTS_DICT.get(experiment):
                da_ini = get_temperature_data(mod, experiment, '2015_2049')
                da_ini.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                da_ini.rio.write_crs("epsg:4326", inplace=True)
                da_ini_clipped = da_ini.copy().rio.clip([india_shape])
                da_ini_clipped = da_ini_clipped.sel(time=slice(f"{period_ini[0]}-01-01", f"{period_ini[1]}-12-31"))
                da_ini_clipped_mean = da_ini_clipped.mean('time')
                
                da_end = get_temperature_data(mod, experiment, '2050_2100')
                da_end.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                da_end.rio.write_crs("epsg:4326", inplace=True)
                da_end_clipped = da_end.copy().rio.clip([india_shape])
                da_end_clipped = da_end_clipped.sel(time=slice(f"{period_end[0]}-01-01", f"{period_end[1]}-12-31"))
                da_end_clipped_mean = da_end_clipped.mean('time')
                
                da_diff = da_end_clipped_mean-da_ini_clipped_mean
                
                
                cmap = plt.get_cmap('RdBu_r')
                vmax = 5
                
                fig,ax = blank_map()
                da_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False,vmin=-vmax,vmax=vmax)
                
                cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
                posn = ax.get_position()
                cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
                norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
                mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                
                cbar_label_var = 'Mean annual temperature difference (°C)'
                _ = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend='neither', extendfrac=0.02)
                
                ax.set_title(f"{mod.upper().replace('_','-')}")
                plt.show()
            pass
        
        # graphe des projections
        if True:
            threshold = 26 #°C
            iea_data = get_cdd_data(threshold=threshold, source='iea_india')
            iea_data = iea_data[iea_data.index.isin(list(range(2011,2025)))]
            
            cdd_projection,_ = export_cdd_INDIA(threshold=threshold)
        
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(iea_data[f'CDD{threshold}'],color='k',label='Historical',zorder=5)
            
            for scenario in EXPERIMENTS_DICT.keys():
                cdd_state_scenario = cdd_projection[cdd_projection.experiment==scenario]
                cdd_state_mean_mod = cdd_state_scenario[['year',f'cdd{threshold}']].groupby('year').mean()
                cdd_state_std_mod = cdd_state_scenario[['year',f'cdd{threshold}']].groupby('year').std()
            
                ax.plot(cdd_state_mean_mod,color=get_ssp_palette(scenario),label=get_ssp_name(scenario))
                
                if scenario in ['ssp1_2_6', 'ssp5_8_5']:
                    ax.fill_between(cdd_state_mean_mod.index,
                                    cdd_state_mean_mod.cdd26+cdd_state_std_mod.cdd26,
                                    cdd_state_mean_mod.cdd26-cdd_state_std_mod.cdd26,
                                    alpha=0.2,color=get_ssp_palette(scenario))
                
            ax.set_ylim(bottom=0.)
            ax.set_xlim(left=2015)
            ax.legend()
            ax.set_ylabel(f'CDD{threshold} (°C.days)')
            plt.show()
        
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()