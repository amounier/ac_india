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
import seaborn as sns
from sklearn.metrics import r2_score

from utils import dict_name_code, draw_state_map, get_ssp_palette

EXPERIMENTS_LIST = ['ssp1','ssp2','ssp3','ssp5']

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


def get_population_projection():
    file = os.path.join('data','iamc_data-3fbc344f-2763-46da-97c3-f4fc654a3f35.csv')
    pop_projection = pd.read_csv(file)
    pop_projection = pop_projection[pop_projection.variable=='Population']
    pop_projection = pop_projection[['scenario',
           '1950', '1955', '1960', '1965', '1970', '1975', '1980', '1985', '1990',
           '1995', '2000', '2005', '2010', '2015', '2020', '2025', '2030', '2035',
           '2040', '2045', '2050', '2055', '2060', '2065', '2070', '2075', '2080',
           '2085', '2090', '2095', '2100']].set_index('scenario').T
    pop_projection.index = pop_projection.index.map(int)
    
    unit = 1e6
    pop_projection = pop_projection*unit
    
    survey_pop = get_population_data()
    
    pop_projection_first_years = pop_projection[pop_projection.index.isin(survey_pop.index)]
    correction_coefficient = survey_pop[survey_pop.index.isin(pop_projection_first_years.index)].sum(axis=1).mean()/pop_projection_first_years['Historical Reference'].mean()
    print(f'Coefficient correcteur : {correction_coefficient:.4f}')
    
    pop_projection = pop_projection*correction_coefficient
    
    pop_projection = pd.DataFrame(index=list(range(1950,2101))).join(pop_projection).interpolate(limit_area='inside')
    
    return pop_projection


def get_gdp_projection():
    file = os.path.join('data','iamc_data-3fbc344f-2763-46da-97c3-f4fc654a3f35.csv')
    pop_projection = pd.read_csv(file)
    pop_projection = pop_projection[pop_projection.variable=='GDP|PPP [per capita]']
    pop_projection = pop_projection[pop_projection.unit=='USD_2017/yr']
    pop_projection = pop_projection[['scenario',
           '1950', '1955', '1960', '1965', '1970', '1975', '1980', '1985', '1990',
           '1995', '2000', '2005', '2010', '2015', '2020', '2025', '2030', '2035',
           '2040', '2045', '2050', '2055', '2060', '2065', '2070', '2075', '2080',
           '2085', '2090', '2095', '2100']].set_index('scenario').T
    pop_projection.index = pop_projection.index.map(int)
    
    unit = 1
    pop_projection = pop_projection*unit
    
    year = 2020
    data = add_gdp_data(base_year=year)
    data = add_pop_data(base_year=year,base_data=data)
    gdp_per_capita_ref = data[f'gdp_{year}'].sum()/data[f'population_{year}'].sum()
    
    
    pop_projection_first_years = pop_projection.loc[year]['Historical Reference']
    calibration_coefficient = gdp_per_capita_ref/pop_projection_first_years
    print(f'Coefficient de calibration : {calibration_coefficient:.4f} rupee/dollar')
    
    pop_projection = pop_projection*calibration_coefficient
    
    pop_projection = pd.DataFrame(index=list(range(1950,2101))).join(pop_projection).interpolate(limit_area='inside')
    
    return pop_projection


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
        
        for var in [f'gdp_{year}',f'population_{year}',f'gdp_per_capita_{year}']:#,f'households_{year}']:
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
    if False:
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
    if False:
        data = add_ac_cooler_data(base_year=2021, base_data=None)
    
        draw_state_map(data.to_dict().get('nb_ac_when_ac_2021'),cbar_min=1,cbar_max=2.2,
                       cbar_label='# of AC systems when equipped (units)',cmap=plt.get_cmap('bone_r'),
                       map_title = '2021',
                       save=os.path.join('maps','nb_ac_when_ac_2021'))
        
        draw_state_map(data.to_dict().get('nb_cooler_when_cooler_2021'),cbar_min=1,cbar_max=2.2,
                       cbar_label='# of cooler systems when equipped (units)',cmap=plt.get_cmap('bone_r'),
                       map_title = '2021',
                       save=os.path.join('maps','nb_cooler_when_cooler_2021'))
        
    #%% Projection de population 
    if True:
        
        pop_projection = get_population_projection()
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(pop_projection['Historical Reference'],color='k',label='Historical',zorder=5)
        # ax.plot(survey_pop.sum(axis=1),color='red',label='Survey',zorder=5)
        
        for scenario in EXPERIMENTS_LIST:
            ax.plot(pop_projection[scenario.upper()],color=get_ssp_palette(scenario),label=scenario.upper())
            
        ax.set_ylim(bottom=0.)
        ax.set_xlim(left=2015)
        ax.legend()
        ax.set_ylabel('Population (inhab)')
        plt.savefig(os.path.join('output','projection_population_india.png'), bbox_inches='tight')
        plt.show()
        
    #%% Projection de PIB par habitant 
    if True:
        
        pop_projection = get_gdp_projection()
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(pop_projection['Historical Reference'],color='k',label='Historical',zorder=5)
        # ax.plot(survey_pop.sum(axis=1),color='red',label='Survey',zorder=5)
        
        for scenario in EXPERIMENTS_LIST:
            ax.plot(pop_projection[scenario.upper()],color=get_ssp_palette(scenario),label=scenario.upper())
            
        ax.set_ylim(bottom=0.)
        ax.set_xlim(left=2015)
        ax.legend()
        ax.set_ylabel('GDP per capita (rupees)')
        plt.savefig(os.path.join('output','projection_gdp_per_capita_india.png'), bbox_inches='tight')
        plt.show()
        
        
    # %% Carcatérisation nombre de ménages par habitant
    if False:
        base_year = 2023
        gdp = add_gdp_data(base_year)
        gdp = add_households_data(base_year,base_data=gdp)
        gdp = add_pop_data(base_year,base_data=gdp)
        gdp[f'ratio_hh_pop_{base_year}'] = gdp[f'households_{base_year}']/gdp[f'population_{base_year}']
        gdp[f'gdp_per_capita_{base_year}'] = gdp[f'gdp_{base_year}']/gdp[f'population_{base_year}']
        
        a,b = np.polyfit(gdp[f'population_{base_year}'],gdp[f'households_{base_year}'],deg=1)
        Y_hat = a*gdp[f'population_{base_year}']+b
        r2 = r2_score(gdp[f'households_{base_year}'],Y_hat)
        
        national_avg = gdp[f'households_{base_year}'].sum()/gdp[f'population_{base_year}'].sum()
        Y_avg = national_avg*gdp[f'population_{base_year}']
        
        # fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        # ax.plot(gdp[f'population_{base_year}'],Y_avg,color='k',label='National average')
        # ax.plot([-1],[0],ls='',alpha=0,label=f'(slope = {national_avg:.2f})')
        # sns.regplot(gdp,x=f'population_{base_year}',y=f'households_{base_year}',ax=ax,
        #             label=f'Linear regression (R$^2$={r2:.2f})',color='tab:blue')
        # ax.plot([-1],[0],ls='',alpha=0,label=f'(slope = {a:.2f})')
        # ax.set_xlabel('Population (inhab)')
        # ax.set_ylabel('Households (unit)')
        # ax.legend()
        # ax.set_title(base_year)
        # ax.set_ylim(bottom=0.)
        # ax.set_xlim(left=0.)
        # plt.show()
        
        a,b = np.polyfit(gdp[f'gdp_per_capita_{base_year}'],gdp[f'ratio_hh_pop_{base_year}'],deg=1)
        Y_hat = a*gdp[f'gdp_per_capita_{base_year}']+b
        r2 = r2_score(gdp[f'gdp_per_capita_{base_year}'],Y_hat)
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        # ax.plot(gdp[f'gdp_per_capita_{base_year}'],gdp[f'ratio_hh_pop_{base_year}'],color='k',marker='o',ls='',label = f'Survey ({base_year})')
        # ax.plot(gdp[f'gdp_per_capita_{base_year}'],Y_hat,label=f'Linear regression (R$^2$={r2:.2f})',color='tab:blue')
        sns.regplot(gdp,x=f'gdp_per_capita_{base_year}',y=f'ratio_hh_pop_{base_year}',ax=ax,
                    label=f'Linear regression (R$^2$={r2:.2f})',color='tab:blue')
        # ax.set_xlabel('Population (inhab)')
        # ax.set_ylabel('Households (unit)')
        ax.legend()
        # ax.set_title(base_year)
        ax.set_ylim(bottom=0.)
        # ax.set_xlim(left=0.)
        plt.show()
        
        
        # pop = get_population_data()
        # hoh = pd.read_excel(os.path.join('data','Households-States-2014-2023.xlsx'),sheet_name='Households',skipfooter=1).set_index('States').T
        
        # ratio_dict = {}
        # for state in hoh.columns:
        #     pop_state = pop[[state]].rename(columns={state:'pop'})
        #     hoh_state = hoh[[state]].rename(columns={state:'hoh'})
        #     pop_state = pop_state.join(hoh_state).dropna()
            
        #     a,b = np.polyfit(pop_state['pop'], pop_state.hoh, deg=1)
        #     ratio_dict[state] = a
        # print()
        # print(f'nb_households = {np.mean(list(ratio_dict.values())):.3f} (+- {np.std(list(ratio_dict.values())):.3f}) * nb_inhab')
    
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()