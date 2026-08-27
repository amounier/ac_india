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

from utils import dict_name_geom
from climate import get_historical_daily_temperature_data
from socioeconomic import get_population_data, get_gdp_data


SMALL_STATES = ['Andaman and Nicobar Islands',
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


def compute_thermal_sensitivity():
    # todo : à coder
    return


#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    daily_temperature = get_historical_daily_temperature_data()
    hourly_consumption = get_historical_hourly_consumption_data()
    daily_consumption = get_historical_daily_consumption_data()
    
    population_data = get_population_data()
    population_data.index = pd.to_datetime([f'{year}-07-01' for year in population_data.index]) 
    
    gdp_data = get_gdp_data()
    gdp_data.index = pd.to_datetime([f'{year}-07-01' for year in gdp_data.index]) 
    
    missing_states_temperature = set([e for e in dict_name_geom.keys() if e not in daily_temperature.columns])
    missing_states_consumption = set([e for e in dict_name_geom.keys() if e not in daily_consumption.columns])
    missing_states_population = set([e for e in dict_name_geom.keys() if e not in population_data.columns])
    missing_states = missing_states_temperature.union(missing_states_consumption,missing_states_population)
    
    # graphe des courbes de demandes
    if False:
        for state in hourly_consumption.columns:
            if state in SMALL_STATES:
                continue 
            
            fig,ax = plt.subplots(figsize=(15,5),dpi=300)
            ax.plot(hourly_consumption[state])
            ax.set_ylim(bottom=0.)
            ax.set_ylabel('Electricity load (MW)')
            ax.set_title(state)
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
    if False:
        calendar = 'IMD'
        normalized = True
        
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
            
            for season,months in seasons_dict.items():
            
                state_data = hourly_consumption[[state]].rename(columns={state:'consumption'})
                if normalized:
                    state_data['consumption'] = state_data.consumption/state_data.consumption.mean()
                    
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
                
                
                ax.plot(week_data.weekday_hour, week_data[('consumption','mean')],
                        label=f'{season}',color=seasons_colors.get(season))
                
            # ax.fill_between(week_data.weekday_hour,
            #                 week_data[('consumption','mean')]+week_data[('consumption','std')],
            #                 week_data[('consumption','mean')]-week_data[('consumption','std')],
            #                 label='_Hourly std',color='tab:blue',alpha=0.33)
            
            if not normalized:
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
    
    # todo consumption par capita per gdp/capita : plutot sur C_base
    # thermosensibilités
    if True:
        for state in daily_consumption.columns:
            if state in missing_states or state in SMALL_STATES: 
                continue 
            
            state_data = (daily_consumption[[state]].rename(columns={state:'consumption'})).join((daily_temperature[[state]].rename(columns={state:'temperature'})),how='inner')
            state_data = state_data.join(population_data[[state]].rename(columns={state:'population'}))
            state_data = state_data.interpolate(limit_area='inside')
            state_data['consumption_per_capita'] = state_data.consumption/state_data.population
                                                     
            state_data['dow'] = state_data.index.dayofweek
            # state_data = state_data[state_data['dow']==6]
                                   
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.scatterplot(state_data,x='temperature',y='consumption_per_capita',color='tab:blue',alpha=0.7)
            ax.plot([-1],[-1],label=f"{state_data.index[0].strftime('%Y/%m/%d')} - {state_data.index[-1].strftime('%Y/%m/%d')}",color='tab:blue',ls='',marker='o')
            # ax.set_ylim(bottom=0.)
            ax.set_xlim([0,40])
            ax.set_ylim([0,0.012])
            ax.set_ylabel('Daily electricity demand per capita (MWh.inhab$^{-1}$)')
            ax.set_xlabel('Daily mean temperature (°C)')
            ax.set_title(f"{state}")
            ax.legend()
            os.makedirs('output/thermal_sensitivity',exist_ok=True)
            plt.savefig(os.path.join(f'output/thermal_sensitivity/thermal_sensitivity_{state}.png'), bbox_inches='tight')
            plt.show()
            plt.close() 
    
    
    
    
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':
    main()