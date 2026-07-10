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

from utils import (save_figure,
                   draw_state_map,
                   dict_name_code,)


def get_cdd_data(threshold=26, source='iea', columns_format_name=True):
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
    if True:
        import cdsapi
        
        
        experiments_dict = {'ssp1_2_6':['cmcc_esm2','iitm_esm','hadgem3_gc31_mm','miroc6','cnrm_cm6_1'],
                            'ssp2_4_5':['cmcc_esm2','iitm_esm','hadgem3_gc31_ll','miroc6','cnrm_cm6_1'],
                            'ssp3_7_0':['cmcc_cm2_sr5','iitm_esm','canesm5','miroc6','cnrm_cm6_1'],
                            'ssp5_8_5':['cmcc_esm2','iitm_esm','hadgem3_gc31_mm','canesm5','miroc6','cnrm_cm6_1'],}
        
        for experiment,models_list in experiments_dict.items():
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
                
                client = cdsapi.Client()
                client.retrieve(dataset, request).download()
                
                downloaded_file = [e for e in os.listdir() if'.zip' in e][0]
                save_file = f'tas_{experiment}_{model}_{years[0]}_{years[-1]}'
                save_path = os.path.join('data','CMIP6')
                subprocess.run(f"unzip {downloaded_file} -d {save_file}",shell=True)
                subprocess.run(f"mv {save_file} {os.path.join(save_path,save_file)}",shell=True)
                subprocess.run(f"rm {downloaded_file}",shell=True)
        
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()