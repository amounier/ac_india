#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:50:46 2026

@author: amounier
"""

import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
# from scipy.optimize import curve_fit
import pwlf
import tqdm

from utils import dict_name_geom, get_ssp_name, get_ssp_palette
from climate import get_historical_daily_temperature_data, format_state_temperature_data_population_weighted, EXPERIMENTS_DICT
from socioeconomic import get_population_data, get_gdp_data, add_ac_cooler_data
from projection import get_ac_projection


SMALL_STATES = ['Andaman and Nicobar Islands',
                'Andaman and Nicobar',
                'Arunachal Pradesh',
                'Assam',
                'Chandigarh',
                'Goa',
                'Jammu and Kashmir',
                'Manipur',
                'Meghalaya',
                'Mizoram',
                'Nagaland',
                'Puducherry',
                'Sikkim',
                'Tripura']

def get_historical_hourly_consumption_data():
    file = os.path.join('data','Indian_load_curve_all_states_2017-2025.csv')
    data = pd.read_csv(file)
    data['datetime'] = pd.to_datetime(data.datetime)
    data = data.set_index('datetime')
    return data


def get_historical_daily_consumption_data():
    hourly = get_historical_hourly_consumption_data()
    daily = hourly.resample('D').sum()
    return daily    


def piecewise_linear(T, T0, C0, kh, kc):
    res = np.piecewise(T, [T < T0, T>=T0], [lambda T: kh*(T-T0) + C0, lambda T: kc*(T-T0)+C0])
    return res


def compute_thermal_sensitivity(force=False):
    file_name = 'thermal_sensitivity.csv'
    file_path = os.path.join('data',file_name)
    
    if file_name not in os.listdir('data') or force:
        daily_temperature = get_historical_daily_temperature_data()
        daily_consumption = get_historical_daily_consumption_data()
        
        population_data = get_population_data()
        population_data.index = pd.to_datetime([f'{year}-07-01' for year in population_data.index]) 
        
        gdp_data = get_gdp_data()
        gdp_data.index = pd.to_datetime([f'{year}-07-01' for year in gdp_data.index]) 
        
        missing_states_temperature = set([e for e in dict_name_geom.keys() if e not in daily_temperature.columns])
        missing_states_consumption = set([e for e in dict_name_geom.keys() if e not in daily_consumption.columns])
        missing_states_population = set([e for e in dict_name_geom.keys() if e not in population_data.columns])
        missing_states = missing_states_temperature.union(missing_states_consumption,missing_states_population)
        
        first_order_thermal_sensitivity_list = ['Andhra Pradesh',
                                                'Chhattisgarh',
                                                'Karnataka',
                                                'Kerala',
                                                'Odisha',
                                                'Tamil Nadu',
                                                'Telangana']
        
        
        data_thermal_sensitivity = None
        
        for state in tqdm.tqdm(daily_consumption.columns):
            if state in missing_states or state in SMALL_STATES: 
                continue 
            
            if state in first_order_thermal_sensitivity_list:
                continue
                
            data_thermal_sensitivity_state = {'state':[],
                                              'dow':[],
                                              'C_base':[],
                                              'T_base':[],
                                              'heating_slope':[],
                                              'cooling_slope':[],
                                              'r2_score':[]}
            
            state_data = (daily_consumption[[state]].rename(columns={state:'consumption'})).join((daily_temperature[[state]].rename(columns={state:'temperature'})),how='inner')
            state_data = state_data.join(population_data[[state]].rename(columns={state:'population'}))
            state_data = state_data.interpolate(limit_area='inside')
            state_data['consumption_per_capita'] = state_data.consumption/state_data.population
            state_data['consumption_per_capita'] = state_data['consumption_per_capita']*1e3 #MWh -> kWh
            state_data['dow'] = state_data.index.dayofweek
            
            for dow in range(7):
                state_data_dow = state_data[state_data['dow']==dow]
                
                state_data_dow = state_data_dow[['temperature','consumption_per_capita']].dropna()
                x,y = state_data_dow.temperature.values, state_data_dow.consumption_per_capita.values
                model = pwlf.PiecewiseLinFit(x, y)
                breaks = model.fit(2)
                
                T_base = breaks[1]
                C0 = model.predict(T_base)[0]
                r2 = model.r_squared()
                heating_slope, cooling_slope = model.calc_slopes()
                
                data_thermal_sensitivity_state['state'].append(state)
                data_thermal_sensitivity_state['dow'].append(dow)
                data_thermal_sensitivity_state['C_base'].append(C0)
                data_thermal_sensitivity_state['T_base'].append(T_base)
                data_thermal_sensitivity_state['heating_slope'].append(heating_slope)
                data_thermal_sensitivity_state['cooling_slope'].append(cooling_slope)
                data_thermal_sensitivity_state['r2_score'].append(r2)
                
            data_thermal_sensitivity_state = pd.DataFrame().from_dict(data_thermal_sensitivity_state)
            
            if data_thermal_sensitivity is None:
                data_thermal_sensitivity = data_thermal_sensitivity_state.copy()
            else:
                data_thermal_sensitivity = pd.concat([data_thermal_sensitivity,data_thermal_sensitivity_state])
            
            # plot
            x_hat = np.linspace(state_data.temperature.values.min(), state_data.temperature.values.max(), 100)
            C,T,slope_heating,slope_cooling,r2 = data_thermal_sensitivity_state[['C_base','T_base','heating_slope','cooling_slope','r2_score']].mean().values
            y_hat = piecewise_linear(x_hat, T, C, slope_heating, slope_cooling)
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            
            sns.scatterplot(state_data,x='temperature',y='consumption_per_capita',color='tab:blue',alpha=0.7)
            ax.plot([-1],[-1],label=f"{state_data.index[0].strftime('%Y/%m/%d')} - {state_data.index[-1].strftime('%Y/%m/%d')}",color='tab:blue',ls='',marker='o')
            ax.plot(x_hat, y_hat, '-',color='k',label=f'Linear fit (R$^2$ = {r2:.2f})')
            
            ax.set_xlim([0,40])
            ax.set_ylim([0,0.012*1e3])
            ax.set_ylabel('Daily electricity demand per capita (kWh.inhab$^{-1}$)')
            ax.set_xlabel('Daily mean temperature (°C)')
            ax.set_title(f"{state}")
            ax.legend()
            
            os.makedirs('output/thermal_sensitivity',exist_ok=True)
            plt.savefig(os.path.join(f'output/thermal_sensitivity/thermal_sensitivity_{state}.png'), bbox_inches='tight')
            plt.show()
            plt.close() 
                
        
        mean_T_base = data_thermal_sensitivity.T_base.mean()
        
        for state in tqdm.tqdm(daily_consumption.columns):
            if state in missing_states or state in SMALL_STATES: 
                continue 
            
            if state not in first_order_thermal_sensitivity_list:
                continue
                
            data_thermal_sensitivity_state = {'state':[],
                                              'dow':[],
                                              'C_base':[],
                                              'T_base':[],
                                              'heating_slope':[],
                                              'cooling_slope':[],
                                              'r2_score':[]}
            
            state_data = (daily_consumption[[state]].rename(columns={state:'consumption'})).join((daily_temperature[[state]].rename(columns={state:'temperature'})),how='inner')
            state_data = state_data.join(population_data[[state]].rename(columns={state:'population'}))
            state_data = state_data.interpolate(limit_area='inside')
            state_data['consumption_per_capita'] = state_data.consumption/state_data.population
            state_data['consumption_per_capita'] = state_data['consumption_per_capita']*1e3 #MWh -> kWh
            state_data['dow'] = state_data.index.dayofweek
            
            for dow in range(7):
                state_data_dow = state_data[state_data['dow']==dow]
                
                state_data_dow = state_data_dow[['temperature','consumption_per_capita']].dropna()
                x,y = state_data_dow.temperature.values, state_data_dow.consumption_per_capita.values
                
                a,b = np.polyfit(x,y,deg=1)
                
                T_base = min(state_data_dow.temperature.values.min(), mean_T_base)
                C0 = a*T_base+b
                r2 = r2_score(y, a*x+b)
                heating_slope, cooling_slope = 0,a
                
                data_thermal_sensitivity_state['state'].append(state)
                data_thermal_sensitivity_state['dow'].append(dow)
                data_thermal_sensitivity_state['C_base'].append(C0)
                data_thermal_sensitivity_state['T_base'].append(T_base)
                data_thermal_sensitivity_state['heating_slope'].append(heating_slope)
                data_thermal_sensitivity_state['cooling_slope'].append(cooling_slope)
                data_thermal_sensitivity_state['r2_score'].append(r2)
                
            data_thermal_sensitivity_state = pd.DataFrame().from_dict(data_thermal_sensitivity_state)
            
            if data_thermal_sensitivity is None:
                data_thermal_sensitivity = data_thermal_sensitivity_state.copy()
            else:
                data_thermal_sensitivity = pd.concat([data_thermal_sensitivity,data_thermal_sensitivity_state])
            
            # plot
            x_hat = np.linspace(T_base, state_data.temperature.values.max(), 100)
            C,T,slope_heating,slope_cooling,r2 = data_thermal_sensitivity_state[['C_base','T_base','heating_slope','cooling_slope','r2_score']].mean().values
            y_hat = piecewise_linear(x_hat, T, C, slope_heating, slope_cooling)
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            
            sns.scatterplot(state_data,x='temperature',y='consumption_per_capita',color='tab:blue',alpha=0.7)
            ax.plot([-1],[-1],label=f"{state_data.index[0].strftime('%Y/%m/%d')} - {state_data.index[-1].strftime('%Y/%m/%d')}",color='tab:blue',ls='',marker='o')
            ax.plot(x_hat, y_hat, '-',color='k',label=f'Linear fit (R$^2$ = {r2:.2f})')
            
            ax.set_xlim([0,40])
            ax.set_ylim([0,0.012*1e3])
            ax.set_ylabel('Daily electricity demand per capita (kWh.inhab$^{-1}$)')
            ax.set_xlabel('Daily mean temperature (°C)')
            ax.set_title(f"{state}")
            ax.legend()
            
            os.makedirs('output/thermal_sensitivity',exist_ok=True)
            plt.savefig(os.path.join(f'output/thermal_sensitivity/thermal_sensitivity_{state}.png'), bbox_inches='tight')
            plt.show()
            plt.close() 
    
                
        data_thermal_sensitivity.to_csv(file_path,index=False)
        
    data_thermal_sensitivity = pd.read_csv(file_path)
    
    return data_thermal_sensitivity


def get_cooling_slope_parameters(force=False):
    ac_data = add_ac_cooler_data(base_year=2021)
    ac_data = ac_data[['ac_per_household_2021']]
    
    sensitivity = compute_thermal_sensitivity(force=force)
    sensitivity = sensitivity.groupby('state').mean().drop(columns=['dow']).reset_index()
    sensitivity = sensitivity.set_index('state')
    sensitivity = sensitivity.join(ac_data,how='left')
    sensitivity = sensitivity.reset_index()
    
    sensitivity['normalized_cooling_slope'] = sensitivity['cooling_slope']/sensitivity['C_base']
    
    x_variable = 'ac_per_household_2021'
    y_variable = 'cooling_slope'
    
    a,b = np.polyfit(sensitivity[x_variable], sensitivity[y_variable], deg=1)
    # print(a,b)
    # x_hat = np.linspace(sensitivity[x_variable].min(),sensitivity[x_variable].max(),num=100)
    y_hat = a*sensitivity[x_variable] + b
    r2 = r2_score(sensitivity[y_variable], y_hat)
    return a,b,r2


def get_C_base_parameters(force=False):
    gdp_data = get_gdp_data()
    gdp_mean = gdp_data[gdp_data.index.isin(list(range(2017,2025)))].mean().to_dict()
    
    population_data = get_population_data()
    pop_mean = population_data[population_data.index.isin(list(range(2017,2025)))].mean().to_dict()
    
    sensitivity = compute_thermal_sensitivity(force=force)
    sensitivity = sensitivity.groupby('state').mean().drop(columns=['dow']).reset_index()
    sensitivity['gdp'] = [gdp_mean.get(e) for e in sensitivity.state]
    sensitivity['pop'] = [pop_mean.get(e) for e in sensitivity.state]
    sensitivity['gdp_per_capita'] = sensitivity['gdp']/sensitivity['pop']
    
    x_variable = 'gdp_per_capita'
    y_variable = 'C_base'
    
    a,b = np.polyfit(np.log(sensitivity[x_variable]), np.log(sensitivity[y_variable]), deg=1)
    y_hat = a*np.log(sensitivity[x_variable]) + b
    r2 = r2_score(np.log(sensitivity[y_variable]), y_hat)
    
    K = np.exp(b)
    alpha = a
    return alpha,K,r2
    
    
#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()

    # etude des courbes de charge 
    if False:
        # daily_temperature = get_historical_daily_temperature_data()
        hourly_consumption = get_historical_hourly_consumption_data()
        # daily_consumption = get_historical_daily_consumption_data()
        
        population_data = get_population_data()
        population_data.index = pd.to_datetime([f'{year}-07-01' for year in population_data.index]) 
        
        gdp_data = get_gdp_data()
        gdp_data.index = pd.to_datetime([f'{year}-07-01' for year in gdp_data.index]) 
        
        # missing_states_temperature = set([e for e in dict_name_geom.keys() if e not in daily_temperature.columns])
        # missing_states_consumption = set([e for e in dict_name_geom.keys() if e not in daily_consumption.columns])
        # missing_states_population = set([e for e in dict_name_geom.keys() if e not in population_data.columns])
        # missing_states = missing_states_temperature.union(missing_states_consumption,missing_states_population)
        
        # graphe des courbes de demandes
        if False:
            for state in hourly_consumption.columns:
                if state in SMALL_STATES:
                    continue 
                
                fig,ax = plt.subplots(figsize=(15,5),dpi=800)
                ax.plot(hourly_consumption[state],color='tab:blue',lw=0.1)
                ax.set_ylim(bottom=0.)
                ax.set_xlim([pd.to_datetime(f'{y}-01-01') for y in [2017,2026]])
                ax.set_ylabel('Electricity load (MW)')
                ax.set_title(state)
                os.makedirs('output/electricity_timeseries',exist_ok=True)
                plt.savefig(os.path.join(f'output/electricity_timeseries/electricity_timeseries_{state}.png'), bbox_inches='tight')
                plt.show()
                plt.close()
                
                
        # graphe des courbes de charge hebdomadaires moyennes 
        if False:
            for state in hourly_consumption.columns:
                if state in SMALL_STATES:
                    continue 
                
                state_data = hourly_consumption[[state]].rename(columns={state:'consumption'})
                state_data['dow'] = state_data.index.dayofweek
                state_data['hour'] = state_data.index.hour
                
                week_data = state_data.groupby(['dow','hour']).agg(func=['mean','std'])
                
                dayofweek_dict = {0:'Mon',
                                  1:'Tue',
                                  2:'Wed',
                                  3:'Thu',
                                  4:'Fri',
                                  5:'Sat',
                                  6:'Sun'}
                
                week_data['weekday_hour'] = [hour + 24*dow for dow,hour in week_data.index]
                
                fig,ax = plt.subplots(figsize=(7.5,5),dpi=300)
                
                ax.plot(week_data.weekday_hour, week_data[('consumption','mean')],
                        label=f'Hourly mean ({state_data.index[0].year}-{state_data.index[-1].year})',color='tab:blue')
                ax.fill_between(week_data.weekday_hour,
                                week_data[('consumption','mean')]+week_data[('consumption','std')],
                                week_data[('consumption','mean')]-week_data[('consumption','std')],
                                label='_Hourly std',color='tab:blue',alpha=0.33)
                
                ax.set_ylim(bottom=0.)
                ylims = ax.get_ylim()
                for e in range(1,7):
                    ax.plot([e*24]*2,ylims,color='k',ls=':',zorder=-1)
                ax.set_ylim(ylims)
                xticks = list(range(0,week_data['weekday_hour'].max()+6,6))
                ax.set_xlim([0,24*7])
                ax.set_xticks(xticks)
                ax.set_xticklabels([f'{e%24}\n ' if e%24!=12 else '12\n{}'.format(dayofweek_dict.get(e//24)) for e in xticks])
                ax.set_ylabel('Electricity load (MW)')
                ax.set_title(state)
                
                ax.legend()
                # if figs_folder is not None:
                #     plt.savefig(os.path.join(figs_folder,'{}.png'.format('{}_heating_cooling_rules'.format(self.name))),bbox_inches='tight')
                plt.show()
                
                
        # graphe des courbes de charge hebdomadaires moyennes avec différence entre les saisons IMD ou Ritu
        if True:
            # calendar = 'IMD'
            calendar = 'CEA'
            normalized = True
            
            df_weekly_factor = {'state':[],
                                'season':[],
                                'dow':[],
                                'hour':[],
                                'intraday_factor':[]}
            
            for state in hourly_consumption.columns:
                if state in SMALL_STATES:
                    continue 
                
                
                fig,ax = plt.subplots(figsize=(7.5,5),dpi=300)
                
                # https://www.agdc.ac.in/pdf/seasons-India_200421.pdf
                if calendar=='IMD':
                    seasons_dict = {'Winter':[12,1,2],
                                    'Summer':[3,4,5],
                                    'Monsoon':[6,7,8,9],
                                    'Post-monsoon':[10,11]}
                    seasons_colors = {'Winter':'tab:blue',
                                      'Summer':'tab:orange',
                                      'Monsoon':'tab:green',
                                      'Post-monsoon':'tab:red'}
                    
                if calendar=='CEA':
                    seasons_dict = {'Winter':[12,1],
                                    'Spring':[2,3],
                                    'Summer':[4,5,6],
                                    'Monsoon':[7,8,9],
                                    'Autumn':[10,11]}
                    
                    cmap = plt.get_cmap('twilight_shifted')
                    seasons_colors = {k:cmap((i+1)/(len(seasons_dict)+0)) for i,k in enumerate(seasons_dict.keys())}
                    # seasons_ls = {k:['solid','dotted','dashed','dashdot'][(i+1)%4] for i,k in enumerate(seasons_dict.keys())} 
                
                for season,months in seasons_dict.items():
                
                    state_data = hourly_consumption[[state]].rename(columns={state:'consumption'})
                    # if normalized:
                    #     state_data['consumption'] = state_data.consumption/state_data.consumption.mean()
                        
                    state_data = state_data[(state_data.index.month).isin(months)]
                    state_data['dow'] = state_data.index.dayofweek
                    state_data['hour'] = state_data.index.hour
                    
                    week_data = state_data.groupby(['dow','hour']).agg(func=['mean','std'])
                    
                    dayofweek_dict = {0:'Mon',
                                      1:'Tue',
                                      2:'Wed',
                                      3:'Thu',
                                      4:'Fri',
                                      5:'Sat',
                                      6:'Sun'}
                    
                    week_data['weekday_hour'] = [hour + 24*dow for dow,hour in week_data.index]
                    if normalized:
                        week_data[('consumption','mean')] = week_data[('consumption','mean')]/week_data[('consumption','mean')].sum()*7
                    
                    df_weekly_factor['state'] += [state]*len(week_data)
                    df_weekly_factor['season'] += [season]*len(week_data)
                    df_weekly_factor['dow'] += list(week_data.index.get_level_values('dow'))
                    df_weekly_factor['hour'] += list(week_data.index.get_level_values('hour'))
                    df_weekly_factor['intraday_factor'] += list(week_data[('consumption','mean')])
                    
                    
                    ax.plot(week_data.weekday_hour, week_data[('consumption','mean')],
                            label=f'{season}',color=seasons_colors.get(season))
                    
                # ax.fill_between(week_data.weekday_hour,
                #                 week_data[('consumption','mean')]+week_data[('consumption','std')],
                #                 week_data[('consumption','mean')]-week_data[('consumption','std')],
                #                 label='_Hourly std',color='tab:blue',alpha=0.33)
                
                # if not normalized:
                ax.set_ylim(bottom=0.,top=0.062)
                    
                ylims = ax.get_ylim()
                for e in range(1,7):
                    ax.plot([e*24]*2,ylims,color='k',ls=':',zorder=-1)
                ax.set_ylim(ylims)
                xticks = list(range(0,week_data['weekday_hour'].max()+6,6))
                ax.set_xlim([0,24*7])
                ax.set_xticks(xticks)
                ax.set_xticklabels([f'{e%24}\n ' if e%24!=12 else '12\n{}'.format(dayofweek_dict.get(e//24)) for e in xticks])
                ax.set_ylabel('Intraday factor (ratio)')
                ax.set_title(state)
                
                ax.legend()
                os.makedirs('output/weekly_consumption',exist_ok=True)
                plt.savefig(os.path.join(f'output/weekly_consumption/weekly_consumption_factor_{state}.png'), bbox_inches='tight')
                plt.show()
            
            if normalized:
                df_weekly_factor = pd.DataFrame().from_dict(df_weekly_factor)
                df_weekly_factor.to_csv(os.path.join('data','weekly_factor.csv'),index=False)
    
#%%============================================================================
# thermosensibilités
# =============================================================================

    # etude des thermosensibilités 
    if False:
        sensitivity = compute_thermal_sensitivity(force=False)
        sensitivity = sensitivity.groupby('state').mean().drop(columns=['dow']).reset_index()
        
        # lien entre la consommation de base par habitant et le PIB par habitant
        if True:
            force = False
            
            gdp_data = get_gdp_data()
            gdp_mean = gdp_data[gdp_data.index.isin(list(range(2017,2025)))].mean().to_dict()
            
            population_data = get_population_data()
            pop_mean = population_data[population_data.index.isin(list(range(2017,2025)))].mean().to_dict()
            
            sensitivity = compute_thermal_sensitivity(force=force)
            sensitivity = sensitivity.groupby('state').mean().drop(columns=['dow']).reset_index()
            sensitivity['gdp'] = [gdp_mean.get(e) for e in sensitivity.state]
            sensitivity['pop'] = [pop_mean.get(e) for e in sensitivity.state]
            sensitivity['gdp_per_capita'] = sensitivity['gdp']/sensitivity['pop']
            
            alpha,K,r2 = get_C_base_parameters(force=False)
            print(f'alpha = {alpha:.3f}, K1 = {K:.3f} (R2 = {r2:.3f})')
            
            x_variable = 'gdp_per_capita'
            y_variable = 'C_base'
            
            x_hat = np.linspace(sensitivity[x_variable].min(),sensitivity[x_variable].max(),num=100)
            
            fig,ax = plt.subplots(dpi=300,figsize=(5,5))
            sns.scatterplot(sensitivity,x=x_variable,y=y_variable,ax=ax,)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel('Base electricity demand per capita (kWh.inhab$^{-1}$)')
            ax.set_xlabel('GDP per capita (rupee.inhab$^{-1}$)')
            ax.plot(x_hat,K*x_hat**alpha,label=f'Linear fit (R$^2$ = {r2:.2f})',color='k')
            ax.legend()
            
            plt.savefig(os.path.join('output/C_base_GDP_per_capita.png'), bbox_inches='tight')
            plt.show()
            plt.close() 
            
            
        # lien entre les pentes et le taux d'équipement
        if True:
            force = False
            
            ac_data = add_ac_cooler_data(base_year=2021)
            ac_data = ac_data[['ac_per_household_2021']]
            
            sensitivity = compute_thermal_sensitivity(force=force)
            sensitivity = sensitivity.groupby('state').mean().drop(columns=['dow']).reset_index()
            sensitivity = sensitivity.set_index('state')
            sensitivity = sensitivity.join(ac_data,how='left')
            sensitivity = sensitivity.reset_index()
            
            a,b,r2 = get_cooling_slope_parameters(force=False)
            print(f'beta = {a:.3f}, K2 = {b:.3f} (R2 = {r2:.3f})')
            
            x_variable = 'ac_per_household_2021'
            y_variable = 'cooling_slope'
            
            np.linspace(sensitivity[x_variable].min(),sensitivity[x_variable].max(),num=100)
            x_hat = np.linspace(sensitivity[x_variable].min(),sensitivity[x_variable].max(),num=100)
            
            fig,ax = plt.subplots(dpi=300,figsize=(5,5))
            sns.scatterplot(sensitivity,x=x_variable,y=y_variable,ax=ax,)
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            ax.set_ylabel('Cooling slope (kWh.inhab$^{-1}$.°C$^{-1}$)')
            ax.set_xlabel('AC equipment rate (ratio)')
            ax.plot(x_hat,a*x_hat+b,label=f'Linear fit (R$^2$ = {r2:.2f})',color='k')
            ax.legend()
            
            plt.savefig(os.path.join('output/cooling_slope_ac_rate.png'), bbox_inches='tight')
            plt.show()
            plt.close() 
            
#%% ===========================================================================
# Projections de consommations
# =============================================================================
    
    # projections journalières 
    if False:
        # détermination des paramètres annuels 
        print('Détermination des paramètres annuels...')
        
        # données thermosensibles
        gdp_data = get_gdp_data()
        gdp_mean = gdp_data[gdp_data.index.isin(list(range(2017,2025)))].mean().to_dict()
        
        population_data = get_population_data()
        pop_mean = population_data[population_data.index.isin(list(range(2017,2025)))].mean().to_dict()
        
        sensitivity = compute_thermal_sensitivity(force=False)
        sensitivity = sensitivity.groupby('state').mean().drop(columns=['dow']).reset_index()
        sensitivity['gdp'] = [gdp_mean.get(e) for e in sensitivity.state]
        sensitivity['pop'] = [pop_mean.get(e) for e in sensitivity.state]
        sensitivity['gdp_per_capita'] = sensitivity['gdp']/sensitivity['pop']
        
        annual_projection = get_ac_projection()
        annual_projection[~annual_projection.state.isin(SMALL_STATES)]
        
        # heating slope (cste)
        heating_slope_dict = sensitivity.set_index('state')['heating_slope'].to_dict()
        annual_projection['heating_slope'] = [heating_slope_dict.get(e) for e in annual_projection.state]
        
        # T_base (cste)
        T_base_dict = sensitivity.set_index('state')['T_base'].to_dict()
        annual_projection['T_base'] = [T_base_dict.get(e) for e in annual_projection.state]
        
        # cooling_slope (f(tau_ac))
        a,_,_ = get_cooling_slope_parameters(force=False)
        
        def compute_cooling_slope(tau_ac, tau_ac_init, cooling_slope_init,a):
            if np.isnan(tau_ac) or np.isnan(tau_ac_init) or np.isnan(cooling_slope_init):
                return np.nan
            return a * (tau_ac-tau_ac_init) + cooling_slope_init
            # return cooling_slope_init # todo : pour analyse de sensibilité
        
        cooling_slope_dict = sensitivity.set_index('state')['cooling_slope'].to_dict()
        ac_rate_dict = add_ac_cooler_data(base_year=2021)['ac_per_household_2021'].to_dict()
        annual_projection['cooling_slope'] = [compute_cooling_slope(tau_ac,ac_rate_dict.get(state,np.nan),cooling_slope_dict.get(state,np.nan),a) for tau_ac,state in zip(annual_projection.ac_equipment_rate,annual_projection.state)]
        
        # test graphique
        if False:
            state = 'Rajasthan'
            experiment = 'ssp3_7_0'
            
            test = annual_projection[(annual_projection.state==state)&
                                     (annual_projection.experiment==experiment)]
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.lineplot(test,x='year',y='cooling_slope',hue='model',ax=ax)
            ax.set_ylabel('Cooling slope (kWh.inhab$^{-1}$.°C$^{-1}$)')
            ax.set_title(f'{state} ({get_ssp_name(experiment)})')
            ax.set_xlabel('')
            plt.show()
            
        # C_base (f(GDP/inhab))
        alpha,_,_ = get_C_base_parameters(force=False)
        
        def compute_C_base(GDP_per_capita, GDP_per_capita_init, C_base_init,alpha):
            if np.isnan(GDP_per_capita) or np.isnan(GDP_per_capita_init) or np.isnan(C_base_init):
                return np.nan
            return (GDP_per_capita/GDP_per_capita_init)**alpha*C_base_init
            # return C_base_init # todo : pour analyse de sensibilité
        
        C_base_dict = sensitivity.set_index('state')['C_base'].to_dict()
        gdp_cap_dict = sensitivity.set_index('state')['gdp_per_capita'].to_dict()
        annual_projection['C_base'] = [compute_C_base(gdp_cap,gdp_cap_dict.get(state,np.nan),C_base_dict.get(state,np.nan),a) for gdp_cap,state in zip(annual_projection.gdp_per_capita,annual_projection.state)]
        
        # test graphique
        if False:
            state = 'Rajasthan'
            model = 'cmcc_esm2'
            
            test = annual_projection[(annual_projection.state==state)&
                                     (annual_projection.model==model)]
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.lineplot(test,x='year',y='C_base',hue='experiment',ax=ax)
            ax.set_ylabel('Base electricity demand per capita (kWh.inhab$^{-1}$)')
            ax.set_title(f'{state} ({model.upper()})')
            ax.set_xlabel('')
            ax.set_ylim(bottom=0.)
            plt.show()
        
        annual_projection['month'] = ['01']*len(annual_projection)
        annual_projection['day'] = ['01']*len(annual_projection)
        annual_projection['date'] = pd.to_datetime(annual_projection[['year','month','day']],format='%Y-%m-%d')
        annual_projection = annual_projection.drop(columns=['year','month','day'])
        annual_projection = annual_projection.set_index(['date','state','experiment','model'])
        
        annual_projection = annual_projection[['population','T_base','C_base','heating_slope','cooling_slope']]
        # print(annual_projection.columns,'\n')
            
        # calcul des consommations journalières 
        print('Calcul des consommations journalières...')
        
        # recalage des temperatures projetées  
        daily_projection = format_state_temperature_data_population_weighted()
        daily_projection = daily_projection[~daily_projection.state.isin(SMALL_STATES)]
        proj_years = list(range(daily_projection.index[0].year,daily_projection.index[-1].year+1))
        
        historical_temperature = get_historical_daily_temperature_data()
        hist_years = list(range(historical_temperature.index[0].year,historical_temperature.index[-1].year+1))
        overlapped_years = sorted(list(set(proj_years).intersection(set(hist_years))))
        
        # déterminations des paramètres de recalage
        calibration_file = 'temperature_calibration.csv'
        
        if calibration_file not in os.listdir('data'):
            
            df_calibration = {'state':[],
                              'experiment':[],
                              'model':[],
                              'multiplier':[],
                              'shifter':[]}
            
            for state in tqdm.tqdm(daily_projection.state.unique()):
                if state not in historical_temperature.columns:
                    print(f'{state} not in historical data...')
                    continue
                
                hist_temp = historical_temperature[historical_temperature.index.year.isin(overlapped_years)][[state]].rename(columns={state:'temperature'})
                hist_temp['cdf'] = hist_temp.rank(method = 'average', pct = True)
                
                for experiment, models in EXPERIMENTS_DICT.items():
                    for model in models:
                        proj_temp = daily_projection[(daily_projection.index.year.isin(overlapped_years))&
                                                     (daily_projection.experiment==experiment)&
                                                     (daily_projection.model==model)&
                                                     (daily_projection.state==state)][['temperature']]
                        
                        proj_temp['cdf'] = proj_temp[['temperature']].rank(method = 'average', pct = True)
                        
                        multiplier = (hist_temp.temperature.max()-hist_temp.temperature.min())/(proj_temp.temperature.max()-proj_temp.temperature.min())
                        shifter = hist_temp.temperature.min() - multiplier*proj_temp.temperature.min()
                        
                        df_calibration['state'].append(state)
                        df_calibration['experiment'].append(experiment)
                        df_calibration['model'].append(model)
                        df_calibration['multiplier'].append(multiplier)
                        df_calibration['shifter'].append(shifter)
                        
                        
            df_calibration = pd.DataFrame().from_dict(df_calibration)
            df_calibration.to_csv(os.path.join('data',calibration_file),index=False)
        
        else:
            df_calibration = pd.read_csv(os.path.join('data',calibration_file))
            df_calibration = df_calibration.set_index(['state','experiment','model'])
            
        
        # application des paramètres de calibration
        daily_projection = daily_projection.reset_index().set_index(['date','state','experiment','model'])
        daily_projection = daily_projection.join(df_calibration, on=['state','experiment','model'])
        daily_projection['calibrated_temperature'] = daily_projection.temperature * daily_projection.multiplier + daily_projection.shifter
        
        
        # calcul des consommations journalières par habitant
        daily_projection = daily_projection.join(annual_projection)
        daily_projection = daily_projection.interpolate(method='linear',limit_area='inside')
        
        def piecewise_linear_array(T, T0, C0, kh, kc):
            T, T0, C0, kh, kc = np.broadcast_arrays(T, T0, C0, kh, kc)
            return np.where(T < T0, kh * (T - T0) + C0, kc * (T - T0) + C0)

        daily_projection['consumption_per_capita'] = piecewise_linear_array(daily_projection.calibrated_temperature, 
                                                                            daily_projection.T_base,
                                                                            daily_projection.C_base, 
                                                                            daily_projection.heating_slope, 
                                                                            daily_projection.cooling_slope)
        
        daily_projection['consumption'] = daily_projection.consumption_per_capita * daily_projection.population * 1e-9 #kWh -> TWh
        
        # test graphique
        if False:
            state = 'Rajasthan'
            model = 'cmcc_esm2'
            
            daily_projection = daily_projection.reset_index()
            
            test = daily_projection[(daily_projection.state==state)&
                                    (daily_projection.model==model)]
            
            # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # for experiment,models in EXPERIMENTS_DICT.items():
            #     if model not in models:
            #         continue
            #     sns.lineplot(test[test.experiment==experiment].groupby(test.date.dt.year)['temperature'].mean().reset_index(),
            #                  x='date',y='temperature',label=get_ssp_name(experiment),
            #                  ax=ax,color=get_ssp_palette(experiment))
            # ax.set_ylabel('Mean annual temperature (°C)')
            # ax.set_title(f'{state} ({model.upper()})')
            # ax.set_xlabel('')
            # ax.legend()
            # ax.set_xlim([2020,2070])
            # plt.show()
            
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for experiment,models in EXPERIMENTS_DICT.items():
                if model not in models:
                    continue
                sns.lineplot(test[test.experiment==experiment].groupby(test.date.dt.year)['consumption_per_capita'].sum().reset_index(),
                             x='date',y='consumption_per_capita',label=get_ssp_name(experiment),
                             ax=ax,color=get_ssp_palette(experiment))
            ax.set_ylabel('Annual electricity consumption per capita (kWh.inhab$^{-1}$)')
            ax.set_title(f'{state} ({model.upper()})')
            ax.set_xlabel('')
            ax.set_xlim([2020,2070])
            ax.set_ylim(bottom=0.)
            ax.legend()
            plt.show()
            
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for experiment,models in EXPERIMENTS_DICT.items():
                if model not in models:
                    continue
                sns.lineplot(test[test.experiment==experiment].groupby(test.date.dt.year)['consumption'].sum().reset_index(),
                             x='date',y='consumption',label=get_ssp_name(experiment),
                             ax=ax,color=get_ssp_palette(experiment))
            ax.set_ylabel('Annual electricity consumption (MWh)')
            ax.set_title(f'{state} ({model.upper()})')
            ax.set_xlabel('')
            ax.set_xlim([2020,2070])
            ax.set_ylim(bottom=0.)
            ax.legend()
            plt.show()
            
            daily_projection = daily_projection.set_index(['date','state','experiment','model'])
        
        daily_projection = daily_projection[~daily_projection.consumption.isna()]
        daily_projection = daily_projection[daily_projection.index.get_level_values('date').year<=2071]
        daily_projection = daily_projection[['temperature','calibrated_temperature','population','consumption']]
        daily_projection.to_parquet(os.path.join('data','consumption.parquet'))

    # sorties graphiques
    if True:
        # consumption : TWh
        # consumption_per_capita : kWh.inhab-1
        daily_projection = pd.read_parquet(os.path.join('data','consumption.parquet')) 
        daily_projection = daily_projection.reset_index()
        daily_projection = daily_projection[daily_projection.date.dt.year>=2020]
        daily_projection = daily_projection[daily_projection.date.dt.year<=2070]
        
        # historical cosnumption (MWh)
        daily_consumption = get_historical_daily_consumption_data()
        daily_consumption = daily_consumption[[s for s in set(daily_projection.state) if s in daily_consumption.columns]]
        annual_consumption = daily_consumption.groupby(daily_consumption.index.year).sum()
        
        daily_projection = daily_projection[daily_projection.state.isin(daily_consumption.columns)]
        
        # graphe national de consommmation électrique
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(annual_consumption.index,annual_consumption.sum(axis=1).values*1e-6,color='k',label='Historical',zorder=5)
        
        for experiment in EXPERIMENTS_DICT.keys():
            daily_projection_exp = daily_projection[daily_projection.experiment==experiment].drop(columns=['experiment'])
            daily_projection_exp = daily_projection_exp[['date','model','consumption']].groupby(['date','model']).sum().reset_index()
            daily_projection_exp = daily_projection_exp.groupby([daily_projection_exp.date.dt.year,'model'])['consumption'].sum().reset_index()
            
            daily_projection_exp_mean_mod = daily_projection_exp[['date','consumption']].groupby('date').mean()
            daily_projection_exp_std_mod = daily_projection_exp[['date','consumption']].groupby('date').std()
            print(f'{experiment} : {daily_projection_exp_mean_mod.loc[2070].values[0]:.0f}+-{daily_projection_exp_std_mod.loc[2070].values[0]:.0f} TWh')
            
            ax.plot(daily_projection_exp_mean_mod,color=get_ssp_palette(experiment),label=get_ssp_name(experiment))
            
            # if experiment in ['ssp1_2_6', 'ssp5_8_5']:
            ax.fill_between(daily_projection_exp_mean_mod.index,
                            daily_projection_exp_mean_mod.consumption+daily_projection_exp_std_mod.consumption,
                            daily_projection_exp_mean_mod.consumption-daily_projection_exp_std_mod.consumption,
                            alpha=0.2,color=get_ssp_palette(experiment))
            
        ax.set_ylim(bottom=0.)
        ax.set_xlim(left=2017)
        ax.legend()
        ax.set_ylabel('Annual electricity consumption (TWh)')
        plt.savefig(os.path.join('output','projection_electricity_consumption_india.png'), bbox_inches='tight')
        plt.show()
    
    # projections horaires
    if False:
        # consumption : TWh
        daily_projection = pd.read_parquet(os.path.join('data','consumption.parquet')) 
        daily_projection = daily_projection.reset_index()
        daily_projection = daily_projection[daily_projection.date.dt.year>=2020]
        daily_projection = daily_projection[['date','state','experiment','model','consumption']]
            
        weekly_factor = pd.read_csv(os.path.join('output','weekly_factor.csv'))
        calendar = 'CEA'
        
        if calendar=='IMD':
            seasons_dict = {'Winter':[12,1,2],
                            'Summer':[3,4,5],
                            'Monsoon':[6,7,8,9],
                            'Post-monsoon':[10,11]}
            
        if calendar=='CEA':
            seasons_dict = {'Winter':[12,1],
                            'Spring':[2,3],
                            'Summer':[4,5,6],
                            'Monsoon':[7,8,9],
                            'Autumn':[10,11]}
            
        month_to_season = {m:k for k,v in seasons_dict.items() for m in v}
            
        output_folder = os.path.join('data','hourly_consumption_projection')
        os.makedirs(output_folder,exist_ok=True)
    
        hourly_projection = None
        
        for state in tqdm.tqdm(daily_projection.state.unique()):
            weekly_factor_state = weekly_factor[weekly_factor.state==state].set_index(['season','dow','hour'])[['intraday_factor']]
            
            for experiment,models in EXPERIMENTS_DICT.items():
                for model in models:
                    output_file = f'hourly_consumption__2020_2070__{state}__{experiment}__{model}.csv'
                    
                    if output_file not in os.listdir(output_folder) and 'hourly_consumption.parquet' not in os.listdir('data'):
                        filtered_daily = daily_projection[(daily_projection.state==state)&
                                                          (daily_projection.experiment==experiment)&
                                                          (daily_projection.model==model)].set_index('date')[['consumption']]
                        
                        filtered_hourly = pd.DataFrame(index=pd.date_range(start=pd.to_datetime('2020-01-01'),end=pd.to_datetime('2071-01-01'),freq='h'))
                        filtered_hourly = filtered_hourly.join(filtered_daily)
                        filtered_hourly = filtered_hourly.interpolate(method='zero')
                        filtered_hourly['season'] = filtered_hourly.index.month.map(month_to_season.get)
                        filtered_hourly['dow'] = filtered_hourly.index.day_of_week
                        filtered_hourly['hour'] = filtered_hourly.index.hour
                        
                        filtered_hourly = filtered_hourly.reset_index().rename(columns={'index':'datetime'}).set_index(['season','dow','hour'])
                        filtered_hourly = filtered_hourly.join(weekly_factor_state).reset_index().drop(columns=['season','dow','hour'])
                        
                        filtered_hourly['consumption'] = filtered_hourly.consumption * filtered_hourly.intraday_factor
                        filtered_hourly = filtered_hourly.set_index('datetime')
                        filtered_hourly = filtered_hourly.drop(columns=['intraday_factor'])
                        
                        filtered_hourly.to_csv(os.path.join(output_folder,output_file),index=True)
                        
                        filtered_hourly['state'] = [state]*len(filtered_hourly)
                        filtered_hourly['experiment'] = [experiment]*len(filtered_hourly)
                        filtered_hourly['model'] = [model]*len(filtered_hourly)
                        
                        if hourly_projection is None:
                            hourly_projection = filtered_hourly
                        else:
                            hourly_projection = pd.concat([hourly_projection,filtered_hourly])
        
        if 'hourly_consumption.parquet' not in os.listdir('data'):
            hourly_projection.to_parquet(os.path.join('data','hourly_consumption.parquet'))

#%% ===========================================================================
# mise en page
# =============================================================================
    # latex des annnexes
    if False:
        daily_projection = pd.read_parquet(os.path.join('data','consumption.parquet')) 
        daily_projection = daily_projection.reset_index()
        daily_projection = daily_projection[daily_projection.date.dt.year>=2020]
        daily_projection = daily_projection[daily_projection.date.dt.year<=2070]
        
        # historical cosnumption (MWh)
        daily_consumption = get_historical_daily_consumption_data()
        daily_consumption = daily_consumption[[s for s in set(daily_projection.state) if s in daily_consumption.columns]]
        annual_consumption = daily_consumption.groupby(daily_consumption.index.year).sum()
        
        daily_projection = daily_projection[daily_projection.state.isin(daily_consumption.columns)]
        
        states = sorted(list(daily_projection.state.unique()))
        
        for idx,state in enumerate(states):
            
            print('\\subsubsection{'+state+'}')
            print()
            print('\\begin{figure}[ht]')
            print('\t\\centering')
            print('\t\\includegraphics[height=0.24\\textheight]{figures/thermal_sensitivity/thermal_sensitivity_'+state+'.png}\\hfill')
            print('\t\\includegraphics[height=0.24\\textheight]{figures/weekly_consumption/weekly_consumption_factor_'+state+'.png}')
            print("\t\\caption{Consommations thermosensibles et hebdomadaires pour "+state+"}")
            print('\\end{figure}')
            print()
            if idx%2 == 1:
                print('\\clearpage')
                print()
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':
    main()