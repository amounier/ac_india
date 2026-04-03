#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:41:49 2026

@author: amounier
"""

import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import dict_name_code, draw_state_map, dict_code_name


def get_gdp_data(columns_format_name=True):
    file = os.path.join('data','GDP-States-2011-2023.xlsx')
    data = pd.read_excel(file,sheet_name='GDP')
    # format state name
    data['States'] = data['States'].str.replace('&','and')
    data = data.set_index('States').transpose()
    data = data * 1e12
    
    if not columns_format_name:
        data = data.rename(columns=dict_name_code)
    return data


def add_gdp_data(base_year=2023, columns_format_name=True, base_data=None):
    data = get_gdp_data(columns_format_name=columns_format_name)
    data = data.loc[base_year]
    data = pd.DataFrame(data).rename(columns={base_year:'gdp'})
    
    data = data.rename(columns={k:f'{k}_{base_year}' for k in data.columns})
    
    if base_data is not None:
        base_data = base_data.join(data)
    else:
        base_data = data
    
    return base_data


def get_population_data(columns_format_name=True):
    file = os.path.join('data','Population-States-2011-2023.xlsx')
    data = pd.read_excel(file,sheet_name='Pop-proj')
    # format state name
    data['States'] = data['States'].str.replace('&','and')
    data = data.set_index('States').transpose()
    data = data * 1e3
    
    if not columns_format_name:
        data = data.rename(columns=dict_name_code)
    return data


def add_pop_data(base_year=2023, columns_format_name=True, base_data=None):
    data = get_population_data(columns_format_name=columns_format_name)
    data = data.loc[base_year]
    data = pd.DataFrame(data).rename(columns={base_year:'population'})
    
    data = data.rename(columns={k:f'{k}_{base_year}' for k in data.columns})
    
    if base_data is not None:
        base_data = base_data.join(data)
    else:
        base_data = data
    return base_data


def add_households_data(base_year=2023, columns_format_name=True, base_data=None):
    file = os.path.join('data','Households-States-2014-2023.xlsx')
    data = pd.read_excel(file,sheet_name='Households',skipfooter=1)
    data = data.set_index('States').T
    
    if not columns_format_name:
        data = data.rename(columns=dict_name_code)
        
    data = data.loc[base_year]
    data = pd.DataFrame(data).rename(columns={base_year:'households'})
    data = data.rename(columns={k:f'{k}_{base_year}' for k in data.columns})
        
    if base_data is not None:
        base_data = base_data.join(data)
    else:
        base_data = data
    return base_data


def get_ac_data(columns_format_name=True):
    # attention : ac + cooler
    file = os.path.join('data','AC-coolers-States-2014-2023-final.xlsx')
    data = pd.read_excel(file,sheet_name='AC+coolers')
    
    # survey only in 2014, 2022 and 2023
    for y in range(2011,2024):
        if y in [2014,2022,2023]:
            continue
        data[y] = [np.nan]*len(data)
        
    # format state name
    data['States'] = data['States'].str.replace('&','and')
    data = data.set_index('States').transpose()
    
    if not columns_format_name:
        data = data.rename(columns=dict_name_code)
    return data


def add_ac_data(base_year=2023, columns_format_name=True, base_data=None):
    data = get_ac_data(columns_format_name=columns_format_name)
    data = data.loc[base_year]
    data = pd.DataFrame(data).rename(columns={base_year:'ac_cooler'})
    
    data = data.rename(columns={k:f'{k}_{base_year}' for k in data.columns})
    
    if base_data is not None:
        base_data = base_data.join(data)
    else:
        base_data = data
    return base_data


def add_ac_cooler_data(base_year=2014, columns_format_name=True, base_data=None):
    if base_year == 2014:
        file = os.path.join('data','AC-coolers-States-2014-2023-final.xlsx')
        data = pd.read_excel(file,sheet_name='pct AC 2014',skiprows=1)[[2014,'TOTAL (cooler + AC)','ratio AC for Total in 2014']]
        data = data.rename(columns={2014:'States','TOTAL (cooler + AC)':'ac_cooler_total','ratio AC for Total in 2014':'ac_cooler_ratio'})
        data['States'] = data['States'].str.replace('&','and')
        data = data[~data.States.isna()]
        data = data.set_index('States')
        data = data[['ac_cooler_total','ac_cooler_ratio']]
        
    elif base_year == 2021:
        file = os.path.join('data','Table 22 - AC, cooler.xlsx')
        data = pd.read_excel(file,sheet_name='Sheet3')
        data = data.rename(columns={'State/UT':'States',
                                    'percentage of households reporting possession of air conditioner ':'ac_per_household',
                                    'average number of air conditioner per households reporting possession of air conditioner (0.0)':'nb_ac_when_ac',
                                    'percentage of households reporting possession of air cooler ':'cooler_per_household',
                                    'average number of air cooler per households reporting possession of air cooler (0.0)':'nb_cooler_when_cooler'})
        data = data.set_index('States')
        data['ac_cooler_ratio'] = data['ac_per_household']/(data['ac_per_household']+data['cooler_per_household'])
        data['ac_per_household'] = data['ac_per_household']/100
        data['cooler_per_household'] = data['cooler_per_household']/100
        
    else:
        data = pd.DataFrame()
    
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
    
    #%% Cartes des variables
    if False:
        year = 2021
        
        data = add_gdp_data(base_year=year)
        data = add_pop_data(base_year=year,base_data=data)
        data = add_ac_data(base_year=2023,base_data=data)
        data = add_households_data(base_year=2020,base_data=data)
        data[f'gdp_per_capita_{year}'] = data[f'gdp_{year}']/data[f'population_{year}']
        # data[f'ac_cooler_per_capita_{year}'] = data[f'ac_cooler_{year}']/data[f'population_{year}']
        
        var_label = {f'gdp_{year}':{'cbar_label':'GDP (rupee)',
                            'save_name':f'gdp_{year}',
                            'cmap':'bone_r'},
                     f'gdp_per_capita_{year}':{'cbar_label':'GDP per capita (rupee.inhab$^{-1}$)',
                                       'save_name':f'gdp_per_capita_{year}',
                                       'cmap':'bone_r'},
                     f'population_{year}':{'cbar_label':'Population (inhab)',
                                   'save_name':f'pop_{year}',
                                   'cmap':'bone_r'},
                     f'households_{year}':{'cbar_label':'Households (#)',
                                   'save_name':f'households_{year}',
                                   'cmap':'bone_r'},
                     }
        
        for var in [f'gdp_{year}',f'population_{year}',f'gdp_per_capita_{year}',f'households_{year}']:
            data_dict = data.to_dict().get(var)
            
            draw_state_map(data_dict,automatic_cbar_values=True,
                           cbar_label=var_label.get(var).get('cbar_label'),cmap=plt.get_cmap(var_label.get(var).get('cmap')),
                           map_title = f'{year}',
                           save=os.path.join('maps',var_label.get(var).get('save_name')))
        
        # TODO à remplacer par AC per household
        # draw_state_map(data.to_dict().get(f'ac_cooler_per_capita_{year}'),automatic_cbar_values=True,
        #                cbar_label='AC & cooler equipment rate (ratio)',cmap=plt.get_cmap('bone_r'),
        #                map_title = f'{year}',
        #                save=os.path.join('maps',f'ac_cooler_rate_{year}'))
    
    #%% Ratio cooler et ac
    if False:
        data = add_ac_cooler_data(base_year=2014, base_data=None)
        data = add_ac_cooler_data(base_year=2021, base_data=data)
        
        draw_state_map(data.to_dict().get('ac_cooler_ratio_2014'),automatic_cbar_values=True,
                       cbar_label='AC vs Cooler ratio (ratio)',cmap=plt.get_cmap('bone_r'),
                       map_title = '2014',
                       save=os.path.join('maps','ac_cooler_rate_2014'))
        
        draw_state_map(data.to_dict().get('ac_cooler_ratio_2021'),automatic_cbar_values=True,
                       cbar_label='AC vs Cooler ratio (ratio)',cmap=plt.get_cmap('bone_r'),
                       map_title = '2021',
                       save=os.path.join('maps','ac_cooler_rate_2021'))

    #%% Carte des taux d'équipement en climatisation
    if True:
        year = 2021
        data = add_ac_cooler_data(base_year=year, base_data=None)
        
        draw_state_map(data.to_dict().get(f'ac_per_household_{year}'),cbar_min=0,cbar_max=0.5,
                       cbar_label='AC equipment rate (ratio)',cmap=plt.get_cmap('bone_r'),
                       map_title = f'{year}',
                       save=os.path.join('maps',f'ac_rate_{year}'))
        
        draw_state_map(data.to_dict().get(f'cooler_per_household_{year}'),cbar_min=0,cbar_max=0.5,
                       cbar_label='Cooler equipment rate (ratio)',cmap=plt.get_cmap('bone_r'),
                       map_title = f'{year}',
                       save=os.path.join('maps',f'cooler_rate_{year}'))
        
    #%% Nombre de climatisation par ménage climatisé
    if True:
        data = add_ac_cooler_data(base_year=2021, base_data=None)
    
        draw_state_map(data.to_dict().get('nb_ac_when_ac_2021'),cbar_min=1,cbar_max=2.2,
                       cbar_label='# of AC systems when equipped (units)',cmap=plt.get_cmap('bone_r'),
                       map_title = '2021',
                       save=os.path.join('maps','nb_ac_when_ac_2021'))
        
        draw_state_map(data.to_dict().get('nb_cooler_when_cooler_2021'),cbar_min=1,cbar_max=2.2,
                       cbar_label='# of cooler systems when equipped (units)',cmap=plt.get_cmap('bone_r'),
                       map_title = '2021',
                       save=os.path.join('maps','nb_cooler_when_cooler_2021'))
        
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()