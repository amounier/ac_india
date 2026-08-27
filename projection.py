#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:01:03 2026

@author: amounier
"""

import time
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import numpy as np
import os 

from climate import export_cdd, EXPERIMENTS_DICT
from utils import get_ssp_name, get_ssp_palette, dict_name_geom
from socioeconomic import (add_gdp_data,
                           add_pop_data,
                           add_households_data,
                           add_ac_cooler_data,
                           get_population_projection,
                           get_gdp_projection)
from characterisation import global_function_exp_sigmoid


#%% ===========================================================================
# script
# =============================================================================
def main():
    tic = time.time()
    
    #%% projection de climatisation
    if True:
        cdd_threshold = 26 #°C
        projection,_,missing_states_cdd = export_cdd(cdd_threshold,force=False)
        
        base_year = 2020
        ref = add_gdp_data(base_year=base_year)
        ref = add_pop_data(base_year=base_year,base_data=ref)
        ref[f'gdp_per_capita_{base_year}'] = ref[f'gdp_{base_year}']/ref[f'population_{base_year}']
        ref_gdp_per_capita = ref[f'gdp_per_capita_{base_year}'].to_dict()
        ref_pop = ref[f'population_{base_year}'].to_dict()
        
        missing_states_gdp_pop = set([e for e in dict_name_geom.keys() if e not in ref_pop.keys()])
        
        gdp = get_gdp_projection()
        gdp = gdp/gdp.loc[2020]['Historical Reference']
        
        pop = get_population_projection()
        pop = pop/pop.loc[2020]['Historical Reference']
        
        proj_gdp_per_capita = []
        proj_pop = []
        for year,state,experiment in tqdm.tqdm(zip(projection.year,projection.state,projection.experiment),total=len(projection)):
            if year < 2025:
                exp_format = 'Historical Reference'
            else:
                exp_format = experiment.split('_')[0].upper()
            
            if state not in ref_gdp_per_capita.keys():
                gdp_val = np.nan
            else:
                gdp_val = gdp.loc[year][exp_format] * ref_gdp_per_capita.get(state)
            proj_gdp_per_capita.append(gdp_val)
            
            if state not in ref_pop.keys():
                pop_val = np.nan
            else:
                pop_val = pop.loc[year][exp_format] * ref_pop.get(state)
            proj_pop.append(pop_val)
                    
        projection['gdp_per_capita'] = proj_gdp_per_capita
        projection['population'] = proj_pop
        
        cdd_projection_rolling = None
        for state in tqdm.tqdm(set(projection.state)):
            for experiment in set(projection.experiment):
                cdd_experiment = projection[(projection.state==state)&
                                            (projection.experiment==experiment)]
                
                for model in set(cdd_experiment.model):
                    cdd = cdd_experiment[cdd_experiment.model==model]
                    cdd[f'cdd{cdd_threshold}_rolling'] = cdd[f'cdd{cdd_threshold}'].rolling(10,center=True).mean()
                    cdd['gdp_per_capita_rolling'] = cdd['gdp_per_capita'].rolling(10,center=True).mean()
                    cdd['population_rolling'] = cdd['population'].rolling(10,center=True).mean()
                    
                    if cdd_projection_rolling is None:
                        cdd_projection_rolling = cdd
                    else:
                        cdd_projection_rolling = pd.concat([cdd_projection_rolling,cdd])
        
        projection = cdd_projection_rolling.reset_index().drop(columns=['index'])
                    
        alpha = 1.45e-3
        beta = -2.40e1
        gamma = 1.34e-5
        delta = 3.10e5
        
        proj_ac_rate = []
        for cdd,gdp_per_cap in tqdm.tqdm(zip(projection[f'cdd{cdd_threshold}_rolling'],projection.gdp_per_capita_rolling),total=len(projection)):
            ac_rate = global_function_exp_sigmoid((cdd,gdp_per_cap),alpha,beta,gamma,delta)
            proj_ac_rate.append(ac_rate)
            
        projection['ac_equipment_rate'] = proj_ac_rate
        
        ref_year = 2021
        ref_ac_rate = add_ac_cooler_data(base_year=ref_year, base_data=None)
        ref_ac_rate = add_households_data(2022,base_data=ref_ac_rate)
        ref_ac_rate_INDIA = (ref_ac_rate[f'ac_per_household_{ref_year}']*ref_ac_rate['households_2022']).sum()/ref_ac_rate['households_2022'].sum()
        ref_ac_rate = ref_ac_rate[f'ac_per_household_{ref_year}'].to_dict()
        
        base_year_pop = 2023
        pop_hh = add_households_data(base_year_pop)
        pop_hh = add_pop_data(base_year_pop,base_data=pop_hh)
        pop_hh['ratio_hh_pop'] = pop_hh[f'households_{base_year_pop}']/pop_hh[f'population_{base_year_pop}']
        ratio_hh_pop = pop_hh.ratio_hh_pop.to_dict()
        
        projection['households'] = [pop*ratio_hh_pop.get(state) if ratio_hh_pop.get(state) is not None else np.nan for pop,state in zip(projection.population,projection.state)]
        projection['households_rolling'] = [pop*ratio_hh_pop.get(state) if ratio_hh_pop.get(state) is not None else np.nan for pop,state in zip(projection.population_rolling,projection.state)]
        
        nb_ac_per_equiped_household = 1.23
        projection['ac_systems'] = projection.households_rolling * projection.ac_equipment_rate * nb_ac_per_equiped_household
        
        print()
        print(projection)
        
        projection.to_csv(os.path.join('data','AC_projection.csv'),index=False)
        
        #%% graphe test
        if False:
            test_state = 'Delhi'
            test_experiment = 'ssp5_8_5'
            test_model = 'iitm_esm'
            
            cdd_test = projection[(projection.state==test_state)&
                                  (projection.model==test_model)&
                                  (projection.experiment==test_experiment)]
            
            fig,ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.plot(cdd_test.year, cdd_test[f'cdd{cdd_threshold}'],label=test_model.upper().replace('_','-'))
            ax.plot(cdd_test.year, cdd_test[f'cdd{cdd_threshold}_rolling'],label='Rolling average')
            # ax.plot(cdd_test.year, cdd_test[f'cdd{cdd_threshold}'].rolling(10,center=True).mean(),label='3')
            ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_title(f'{test_state} - {get_ssp_name(test_experiment)}')
            plt.show()
            
            fig,ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.plot(cdd_test.year, cdd_test['gdp_per_capita'],label='OECD ENV-Growth 2025')
            ax.plot(cdd_test.year, cdd_test['gdp_per_capita_rolling'],label='Rolling average')
            # ax.plot(cdd_test.year, cdd_test[f'cdd{cdd_threshold}'].rolling(10,center=True).mean(),label='3')
            ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_title(f'{test_state} - {get_ssp_name(test_experiment)}')
            plt.show()
            
            fig,ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.plot(cdd_test.year, cdd_test['population'],label='IIASA-WiC POP 2025')
            ax.plot(cdd_test.year, cdd_test['population_rolling'],label='Rolling average')
            # ax.plot(cdd_test.year, cdd_test[f'cdd{cdd_threshold}'].rolling(10,center=True).mean(),label='3')
            ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_title(f'{test_state} - {get_ssp_name(test_experiment)}')
            plt.show()
            
            fig,ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.plot([ref_year],[ref_ac_rate.get(test_state)],ls='',marker='o',color='k',zorder=5,label='Survey')
            ax.plot(cdd_test.year, cdd_test['ac_equipment_rate'],label='Model')
            # ax.plot(cdd_test.year, cdd_test[f'cdd{cdd_threshold}'].rolling(10,center=True).mean(),label='3')
            ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_title(f'{test_state} - {get_ssp_name(test_experiment)}')
            plt.show()
        
        
        #%% Graphes propres
        
        # génération du latex
        if True:
            missing = missing_states_cdd.union(missing_states_gdp_pop)
            for state in sorted(list(dict_name_geom.keys())):
                if state in missing:
                    continue
                
                print('\\subsubsection{'+state+'}')
                print()
                print('\\begin{figure}[ht]')
                print('\t\\centering')
                print('\t\\includegraphics[height=0.25\\textheight]{figures/projection_states/projection_cdd26_'+state+'.png}\\hfill')
                print('\t\\includegraphics[height=0.25\\textheight]{figures/projection_states/projection_gdp_per_capita_'+state+'.png}\\\\')
                print('\t\\includegraphics[height=0.25\\textheight]{figures/projection_states/projection_ac_rate_'+state+'.png}\\hfill')
                print('\t\\includegraphics[height=0.25\\textheight]{figures/projection_states/projection_ac_systems_'+state+'.png}')
                print("\t\\caption{Projections des différentes variables pour "+state+"}")
                print('\\end{figure}')
                print()
                print('\\clearpage')
                print()
        

        # graphes par état
        if False:
            print(missing_states_cdd.union(missing_states_gdp_pop))
            
            for test_state in dict_name_geom.keys():
                if test_state in missing_states_cdd or test_state in missing_states_gdp_pop:
                    continue
            
                # graphe évolution GDP_per_capita
                years = list(range(2020,2071,1))
                state_projection = projection[(projection.state==test_state)&(projection.year.isin(years))]
                
                fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                # ax.plot([ref_year],[ref_ac_rate.get(test_state)],ls='',marker='o',color='k',zorder=5,label='Survey')
                
                for idx,experiment in enumerate(EXPERIMENTS_DICT.keys()):
                    state_exp = state_projection[state_projection.experiment==experiment]
                    mean_value = state_exp[['year','gdp_per_capita']].groupby('year').mean()
                    std_value = state_exp[['year','gdp_per_capita']].groupby('year').std()
                    
                    ax.plot(mean_value.index,mean_value.values.T[0],color=get_ssp_palette(experiment),
                            label=get_ssp_name(experiment))
                    
                    if experiment in ['ssp1_2_6', 'ssp5_8_5']:
                        ax.fill_between(mean_value.index,
                                        mean_value.values.T[0]+std_value.values.T[0],
                                        mean_value.values.T[0]-std_value.values.T[0],
                                        alpha=0.2,color=get_ssp_palette(experiment))
                
                ax.legend()
                ax.set_ylim(bottom=0.)
                ax.set_ylabel('GDP per capita (rupees)')
                ax.set_title(f'{test_state}')
                plt.savefig(os.path.join('output','projection_states',f'projection_gdp_per_capita_{test_state}.png'), bbox_inches='tight')
                plt.show()
                
                # graphe evolution CDD
                years = list(range(2020,2071,1))
                
                state_projection = projection[(projection.state==test_state)&(projection.year.isin(years))]
                
                # fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                # # ax.plot([ref_year],[ref_ac_rate.get(test_state)],ls='',marker='o',color='k',zorder=5,label='Survey')
                
                # for idx,experiment in enumerate(EXPERIMENTS_DICT.keys()):
                #     state_exp = state_projection[state_projection.experiment==experiment]
                #     mean_value = state_exp[['year',f'cdd{cdd_threshold}_rolling']].groupby('year').mean()
                #     std_value = state_exp[['year',f'cdd{cdd_threshold}_rolling']].groupby('year').std()
                    
                #     ax.plot(mean_value.index,mean_value.values.T[0],color=get_ssp_palette(experiment),
                #             label=get_ssp_name(experiment))
                    
                #     if experiment in ['ssp1_2_6', 'ssp5_8_5']:
                #         ax.fill_between(mean_value.index,
                #                         mean_value.values.T[0]+std_value.values.T[0],
                #                         mean_value.values.T[0]-std_value.values.T[0],
                #                         alpha=0.2,color=get_ssp_palette(experiment))
                
                # ax.legend()
                # ax.set_ylim(bottom=0.)
                # ax.set_ylabel(f'10-years rolling average CDD{cdd_threshold} (°C.days)')
                # ax.set_title(f'{test_state}')
                # plt.show()
                
                
                # graphe évolution taux équipement par état
                years = list(range(2020,2071,10))
                
                state_projection = projection[(projection.state==test_state)&(projection.year.isin(years))]
                
                fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                ax.plot([ref_year],[ref_ac_rate.get(test_state)],ls='',marker='o',color='k',zorder=5,label='Survey')
                
                for idx,experiment in enumerate(EXPERIMENTS_DICT.keys()):
                    state_exp = state_projection[state_projection.experiment==experiment]
                    mean_ac_rate = state_exp[['year','ac_equipment_rate']].groupby('year').mean()
                    std_ac_rate = state_exp[['year','ac_equipment_rate']].groupby('year').std()
                    
                    ax.errorbar(mean_ac_rate.index+(idx-2)/2,mean_ac_rate.values.T[0],std_ac_rate.values.T[0]*2,
                                capsize=3,color=get_ssp_palette(experiment),
                                label=get_ssp_name(experiment),ls=':',marker='o',mec='w')
                
                ax.legend()
                ax.set_ylim(bottom=0.,top=1.)
                ax.set_ylabel('AC equipment rate (ratio)')
                ax.set_title(f'{test_state}')
                plt.savefig(os.path.join('output','projection_states',f'projection_ac_rate_{test_state}.png'), bbox_inches='tight')
                plt.show()
                
                # graphe évolution nombre équipement par état
                years = list(range(2020,2071,10))
                
                state_projection = projection[(projection.state==test_state)&(projection.year.isin(years))]
                
                fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                
                for idx,experiment in enumerate(EXPERIMENTS_DICT.keys()):
                    state_exp = state_projection[state_projection.experiment==experiment]
                    mean_ac_rate = state_exp[['year','ac_systems']].groupby('year').mean()
                    std_ac_rate = state_exp[['year','ac_systems']].groupby('year').std()
                    
                    ax.errorbar(mean_ac_rate.index+(idx-2)/2,mean_ac_rate.values.T[0],std_ac_rate.values.T[0]*2,
                                capsize=3,color=get_ssp_palette(experiment),
                                label=get_ssp_name(experiment),ls=':',marker='o',mec='w')
                
                ax.legend()
                ax.set_ylim(bottom=0.)
                ax.set_ylabel('AC systems (unit)')
                ax.set_title(f'{test_state}')
                plt.savefig(os.path.join('output','projection_states',f'projection_ac_systems_{test_state}.png'), bbox_inches='tight')
                plt.show()
                
        # graphes nationaux
        if False:
            
            # graphe évolution taux équipement national
            years = list(range(2020,2071,10))
            
            state_projection = projection[projection.year.isin(years)]
            
            fig,ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.plot([ref_year],[ref_ac_rate_INDIA],ls='',marker='o',color='k',zorder=5,label='Survey')
            
            for idx,experiment in enumerate(EXPERIMENTS_DICT.keys()):
                state_exp = state_projection[state_projection.experiment==experiment].dropna()
                state_exp['nb_equiped_households'] = state_exp['ac_equipment_rate'] * state_exp['households_rolling']
                sum_exp = state_exp[['year','model','nb_equiped_households','households_rolling']].groupby(['year','model']).sum()
                sum_exp['ac_equipment_rate'] = sum_exp.nb_equiped_households/sum_exp.households_rolling
                sum_exp['ac_systems'] = sum_exp.nb_equiped_households * nb_ac_per_equiped_household
                
                sum_exp = sum_exp.reset_index()
                
                mean_ac_rate = sum_exp[['year','ac_equipment_rate']].groupby('year').mean()
                std_ac_rate = sum_exp[['year','ac_equipment_rate']].groupby('year').std()
                
                ax.errorbar(mean_ac_rate.index+(idx-2)/2,mean_ac_rate.values.T[0],std_ac_rate.values.T[0]*2,
                            capsize=3,color=get_ssp_palette(experiment),
                            label=get_ssp_name(experiment),ls=':',marker='o',mec='w')
            
            ax.legend()
            ax.set_ylim(bottom=0.,top=1.)
            ax.set_ylabel('AC equipment rate (ratio)')
            ax.set_title('')
            plt.savefig(os.path.join('output','projection_ac_rate_india.png'), bbox_inches='tight')
            plt.show()
            
            # graphe évolution nombre de systèmes INDIA
            years = list(range(2020,2071,10))
            
            state_projection = projection[projection.year.isin(years)]
            
            fig,ax = plt.subplots(dpi=300,figsize=(5,5))
            # ax.plot([ref_year],[ref_ac_rate_INDIA],ls='',marker='o',color='k',zorder=5,label='Survey')
            
            for idx,experiment in enumerate(EXPERIMENTS_DICT.keys()):
                state_exp = state_projection[state_projection.experiment==experiment].dropna()
                state_exp['nb_equiped_households'] = state_exp['ac_equipment_rate'] * state_exp['households_rolling']
                sum_exp = state_exp[['year','model','nb_equiped_households','households_rolling']].groupby(['year','model']).sum()
                sum_exp['ac_equipment_rate'] = sum_exp.nb_equiped_households/sum_exp.households_rolling
                sum_exp['ac_systems'] = sum_exp.nb_equiped_households * nb_ac_per_equiped_household
                
                sum_exp = sum_exp.reset_index()
                
                mean_ac_rate = sum_exp[['year','ac_systems']].groupby('year').mean()
                std_ac_rate = sum_exp[['year','ac_systems']].groupby('year').std()
                
                ax.errorbar(mean_ac_rate.index+(idx-2)/2,mean_ac_rate.values.T[0],std_ac_rate.values.T[0]*2,
                            capsize=3,color=get_ssp_palette(experiment),
                            label=get_ssp_name(experiment),ls=':',marker='o',mec='w')
            
            ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_ylabel('AC systems (unit)')
            ax.set_title('')
            plt.savefig(os.path.join('output','projection_ac_systems_india.png'), bbox_inches='tight')
            
            plt.show()
                
        
    
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':
    main()