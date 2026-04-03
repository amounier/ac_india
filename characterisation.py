#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:30:51 2026

@author: amounier
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os
import pandas as pd
from scipy.special import erf

from climate import add_cdd_data
from socioeconomic import (add_ac_cooler_data, 
                           add_gdp_data, 
                           add_pop_data,
                           add_households_data)



def illustrate_functions(a=1,b=1.5,c=5,mode='Sigmoid',X_max=10):
    if mode == 'Sigmoid':
        X = np.linspace(0,X_max,100)
        Y = 1/(1+a*np.exp(-b*(X-c)))
        
    elif mode == 'Exponential':
        X = np.linspace(0,X_max,100)
        Y = 1-a*np.exp(-b*(X-c))
        
    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
    ax.plot(X,Y,color='k',label=mode)
    ax.legend()
    ax.set_ylim([0,1.])
    ax.set_xlim([X[0],X[-1]])
    plt.show()
    return 


def global_function_sigmoid_sigmoid(X,b,c,e,f):
    CDD = X[0]
    GDP = X[1]
    a = 1
    d = 1
    Y1 = 1/(1+a*np.exp(-b*(CDD-c)))
    Y2 = 1/(1+d*np.exp(-e*(GDP-f)))
    Y = Y1*Y2
    return Y

def global_function_sigmoid_exp(X,b,c,e,f):
    CDD = X[0]
    GDP = X[1]
    a = 1
    d = 1
    Y1 = 1/(1+a*np.exp(-b*(CDD-c)))
    Y2 = 1-d*np.exp(-e*(GDP-f))
    Y = Y1*Y2
    return Y

def global_function_exp_sigmoid(X,b,c,e,f):
    CDD = X[0]
    GDP = X[1]
    a = 1
    d = 1
    Y1 = 1-a*np.exp(-b*(CDD-c))
    Y2 = 1/(1+d*np.exp(-e*(GDP-f)))
    Y = Y1*Y2
    return Y

def global_function_exp_exp(X,b,c,e,f):
    CDD = X[0]
    GDP = X[1]
    a = 1
    d = 1
    Y1 = 1-a*np.exp(-b*(CDD-c))
    Y2 = 1-d*np.exp(-e*(GDP-f))
    Y = Y1*Y2
    
    return Y




def main():
    tic = time.time()
    
    #%% Caractérisation du taux d'équipement
    if False:
        os.makedirs('output/ac_equipment',exist_ok=True)
        
        year = 2021
        cdd_mean_period = 10 #yr

        for fct in [global_function_sigmoid_sigmoid,global_function_exp_sigmoid,global_function_exp_exp,global_function_sigmoid_exp]:
            
            print(fct)
            _,_,cdd_func,gdp_func = str(fct).split(' ')[1].split('_')
            title = f'Function : {cdd_func} $\\times$ {gdp_func}'
            save_title = f'{cdd_func}_{gdp_func}'
            
            res = {'CDD':[],'a':[],'b':[],'c':[],'d':[],'r2':[]}
            for cdd_threshold in [18,21,23,26]:
                
                # print('\t',f'CDD{cdd_threshold}')
        
                data = add_cdd_data(year, mean_period=cdd_mean_period, threshold=cdd_threshold, source='iea', columns_format_name=True,base_data=None)
                data = add_ac_cooler_data(year,columns_format_name=True, base_data=data)
                data = add_gdp_data(base_year=year, base_data=data)
                data = add_pop_data(base_year=year, base_data=data)
                data = add_households_data(base_year=2020, base_data=data)
                data['sqrt_inv_households_2020'] = np.sqrt(1/data.households_2020)*0+1
                data[f'gdp_per_capita_{year}'] = data[f'gdp_{year}']/data[f'population_{year}']
                data = data.reset_index()
                
                var = f'ac_per_household_{year}'
                # var = f'cooler_per_household_{year}'
                
                data = data[[f'cdd{cdd_threshold}_{year}',var,f'gdp_per_capita_{year}','sqrt_inv_households_2020','households_2020']]
                data = data.dropna()
                
                X = data[[f'cdd{cdd_threshold}_{year}',f'gdp_per_capita_{year}']].values.T
                Y = data[var].values
                
                p0 = (8.5/max(X[0]),max(X[0])/3,8.5/max(X[1]),max(X[1])/3)
                try: 
                    popt, _ = curve_fit(fct,X,Y,p0,sigma=data['sqrt_inv_households_2020'].values,absolute_sigma=False,maxfev=100000)
                except RuntimeError:
                    popt = [np.nan]*4
                    continue
                # print('\t\t',popt)
                
                Y_hat = fct(X, *popt)
                r2 = r2_score(Y, Y_hat)
                
                res['CDD'].append(cdd_threshold)
                res['a'].append(popt[0])
                res['b'].append(popt[1])
                res['c'].append(popt[2])
                res['d'].append(popt[3])
                res['r2'].append(r2)
                # print('\t\t\t',r2)
                
                fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,5*0.92),dpi=300)
                sns.scatterplot(data,x=f'cdd{cdd_threshold}_{year}',y=var,ax=ax1,legend=False,label='Data',edgecolors='k',color='lightgrey')
                ax1.plot(X[0],Y_hat,ls='',marker='+',label=f'Model (R$^2$={r2:.2f})',color='grey')
                ax1.set_xlabel(f'CDD{cdd_threshold} in {year-cdd_mean_period+1}-{year} (°C.days)')
                ax1.legend()
                sns.scatterplot(data,x=f'gdp_per_capita_{year}',y=var,ax=ax2,legend=False,edgecolors='k',color='lightgrey')
                ax2.plot(X[1],Y_hat,ls='',marker='+',color='grey')
                ax2.set_xlabel(f'GDP per capita in {year} (rupees)')
                ax1.set_ylabel('AC equipment rate in households (ratio)')
                plt.subplots_adjust(wspace=0.1,top=0.92)
                fig.suptitle(title)
                plt.savefig(os.path.join('output','ac_equipment',f'cdd{cdd_threshold}_{save_title}_scatter.png'), bbox_inches='tight')
                plt.show()
                
                
                # affichage en surface
                if True:
                    nb_discrete = 50
                    CDD = np.linspace(data[[f'cdd{cdd_threshold}_{year}']].min().values[0],data[[f'cdd{cdd_threshold}_{year}']].max().values[0],nb_discrete)
                    GDP = np.linspace(data[[f'gdp_per_capita_{year}']].min().values[0],data[[f'gdp_per_capita_{year}']].max().values[0],nb_discrete)
                    
                    GDP,CDD = np.meshgrid(GDP,CDD)
                    Z = np.zeros(CDD.shape)
                    for i in range(nb_discrete):
                        for j in range(nb_discrete):
                            Z[i,j] = np.clip(fct((CDD[i,j],GDP[i,j]), *popt),a_min=0,a_max=None)
                    
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    contour = ax.contourf(GDP,CDD,Z,vmin=0,vmax=1,cmap='bone_r',levels=np.arange(0.,1.01,step=0.05))
                    ax.set_title(title)
                    ax.set_xlabel(f'GDP per capita in {year} (rupees)')
                    ax.set_ylabel(f'CDD{cdd_threshold} in {year-cdd_mean_period+1}-{year} (°C.days)')
                    ax_cb = fig.add_axes([0,0,0.1,0.1])
                    posn = ax.get_position()
                    ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
                    fig.add_axes(ax_cb)
                    fig.colorbar(contour, cax=ax_cb,)
                    ax_cb.set_ylabel('AC equipment rate in households (ratio)')
                    plt.savefig(os.path.join('output','ac_equipment',f'cdd{cdd_threshold}_{save_title}_2dplot.png'), bbox_inches='tight')
                    plt.show()
            
            res = pd.DataFrame().from_dict(res)
            res = res.set_index('CDD').T
            print(res)
            print(res.to_latex(float_format='%.2e'))
            print()
        
    #%% Caractérisation des nombre de climatisations
    if False:
        os.makedirs('output/ac_equipment',exist_ok=True)
        
        year = 2021
        cdd_mean_period = 10 #yr
        cdd_threshold = 26
        
        data = add_cdd_data(year, mean_period=cdd_mean_period, threshold=cdd_threshold, source='iea', columns_format_name=True,base_data=None)
        data = add_ac_cooler_data(year,columns_format_name=True, base_data=data)
        data = add_gdp_data(base_year=year, base_data=data)
        data = add_pop_data(base_year=year, base_data=data)
        data[f'gdp_per_capita_{year}'] = data[f'gdp_{year}']/data[f'population_{year}']
        data = data.reset_index()
        
        var = f'nb_ac_when_ac_{year}'
        mean = data[var].mean()
        
        data = data[[f'cdd{cdd_threshold}_{year}',var,f'gdp_per_capita_{year}']]
        data = data.dropna()
        
        X = data[f'cdd{cdd_threshold}_{year}'].values
        Y = data[var].values
        
        def exp_func(X,b,c):
            Y1 = 2-np.exp(-b*(X-c))
            Y1 = b*X+c
            return Y1
        
        p0 = (8.5/max(X),0)
        try: 
            popt, _ = curve_fit(exp_func,X,Y,p0,maxfev=100000)
        except RuntimeError:
            popt = [np.nan]*3
        print(popt)
        save = 'nb_ac_when_ac'
        
        Y_hat = exp_func(X, *popt)
        r2 = r2_score(Y, Y_hat)
        
        fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,5*0.92),dpi=300)
        sns.scatterplot(data,x=f'cdd{cdd_threshold}_{year}',y=var,ax=ax1,legend=False,label=f'Data ($\\mu=${mean:.1f})',edgecolors='k',color='lightgrey')
        ax1.plot(X,Y_hat,ls='',marker='+',label=f'Model (R$^2$={r2:.2f})',color='grey')
        ax1.set_xlabel(f'CDD{cdd_threshold} in {year-cdd_mean_period+1}-{year} (°C.days)')
        ax1.legend()
        sns.scatterplot(data,x=f'gdp_per_capita_{year}',y=var,ax=ax2,legend=False,edgecolors='k',color='lightgrey')
        ax2.plot(data[f'gdp_per_capita_{year}'].values,Y_hat,ls='',marker='+',color='grey')
        ax2.set_xlabel(f'GDP per capita in {year} (rupees)')
        ax1.set_ylabel('Number of AC systems in equiped households (#)')
        plt.subplots_adjust(wspace=0.1,top=0.92)
        # fig.suptitle(title)
        plt.savefig(os.path.join('output','ac_equipment',f'cdd{cdd_threshold}_{save}_scatter.png'), bbox_inches='tight')
        plt.show()
        
    #%% Caractérisation des refroidisseurs d'air
    if True:
        os.makedirs('output/cooler_equipment',exist_ok=True)
        
        year = 2021
        cdd_mean_period = 10 #yr
        cdd_threshold = 26
        
        data = add_cdd_data(year, mean_period=cdd_mean_period, threshold=cdd_threshold, source='iea', columns_format_name=True,base_data=None)
        data = add_ac_cooler_data(year,columns_format_name=True, base_data=data)
        data = add_gdp_data(base_year=year, base_data=data)
        data = add_pop_data(base_year=year, base_data=data)
        data[f'gdp_per_capita_{year}'] = data[f'gdp_{year}']/data[f'population_{year}']
        data = data.reset_index()
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        sns.scatterplot(data,x=f'ac_per_household_{year}',y=f'cooler_per_household_{year}',ax=ax,edgecolors='k',color='lightgrey')
        ax.set_ylabel('Cooler equipment rate in households (ratio)')
        ax.set_xlabel('AC equipment rate in households (ratio)')
        ax.set_xlim([-0.02,0.5])
        ax.set_ylim([-0.02,0.5])
        plt.savefig(os.path.join('output','cooler_equipment',f'cooler_ac_scatter_{year}.png'), bbox_inches='tight')
        plt.show()
        
        
        var = f'cooler_per_household_{year}'
        mean = data[var].mean()
        
        data = data[[f'cdd{cdd_threshold}_{year}',var,f'gdp_per_capita_{year}']]
        data = data.dropna()
        
        X = data[f'cdd{cdd_threshold}_{year}'].values
        Y = data[var].values
        
        def exp_func(X,b,c):
            Y1 = 1/(1+np.exp(-b*(X-c)))
            # Y1 = (1+np.exp(-(X-c)))**(-b)
            # Y1 = erf(-b*(X-c))
            # Y1 = b*X+c
            return Y1
        
        p0 = (8.5/max(X),max(X)/3)
        try: 
            popt, _ = curve_fit(exp_func,X,Y,p0,maxfev=100000)
        except RuntimeError:
            popt = [np.nan]*3
        print(popt)
        save = 'cooler_rate'
        
        Y_hat = exp_func(X, *popt)
        r2 = r2_score(Y, Y_hat)
        
        fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,5*0.92),dpi=300)
        sns.scatterplot(data,x=f'cdd{cdd_threshold}_{year}',y=var,ax=ax1,legend=False,label=f'Data',edgecolors='k',color='lightgrey')
        ax1.plot(X,Y_hat,ls='',marker='+',label=f'Model (R$^2$={r2:.2f})',color='grey')
        ax1.set_xlabel(f'CDD{cdd_threshold} in {year-cdd_mean_period+1}-{year} (°C.days)')
        ax1.legend()
        sns.scatterplot(data,x=f'gdp_per_capita_{year}',y=var,ax=ax2,legend=False,edgecolors='k',color='lightgrey')
        ax2.plot(data[f'gdp_per_capita_{year}'].values,Y_hat,ls='',marker='+',color='grey')
        ax2.set_xlabel(f'GDP per capita in {year} (rupees)')
        ax1.set_ylabel('Cooler equipment rate in households (ratio)')
        plt.subplots_adjust(wspace=0.1,top=0.92)
        # fig.suptitle(title)
        plt.savefig(os.path.join('output','cooler_equipment',f'cdd{cdd_threshold}_{save}_scatter.png'), bbox_inches='tight')
        plt.show()
    
    # Nombre de refroidisseurs 
    if True:
        year = 2021
        cdd_mean_period = 10 #yr
        cdd_threshold = 26
        
        data = add_cdd_data(year, mean_period=cdd_mean_period, threshold=cdd_threshold, source='iea', columns_format_name=True,base_data=None)
        data = add_ac_cooler_data(year,columns_format_name=True, base_data=data)
        data = add_gdp_data(base_year=year, base_data=data)
        data = add_pop_data(base_year=year, base_data=data)
        data[f'gdp_per_capita_{year}'] = data[f'gdp_{year}']/data[f'population_{year}']
        data = data.reset_index()
        
        var = f'nb_cooler_when_cooler_{year}'
        mean = data[var].mean()
        
        data = data[[f'cdd{cdd_threshold}_{year}',var,f'gdp_per_capita_{year}']]
        data = data.dropna()
        
        X = data[f'gdp_per_capita_{year}'].values
        Y = data[var].values
        
        def exp_func(X,b,c):
            Y1 = 1+np.exp(-b*(X-c))
            # Y1 = b*X+c
            return Y1
        
        p0 = (8.5/max(X),0)
        try: 
            popt, _ = curve_fit(exp_func,X,Y,p0,maxfev=100000)
        except RuntimeError:
            popt = [np.nan]*3
        print(popt)
        save = 'nb_cooler_when_cooler'
        
        Y_hat = exp_func(X, *popt)
        r2 = r2_score(Y, Y_hat)
        
        fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,5*0.92),dpi=300)
        sns.scatterplot(data,x=f'cdd{cdd_threshold}_{year}',y=var,ax=ax1,legend=False,label=f'Data ($\\mu=${mean:.1f})',edgecolors='k',color='lightgrey')
        ax1.plot(data[f'cdd{cdd_threshold}_{year}'].values,Y_hat,ls='',marker='+',label=f'Model (R$^2$={r2:.2f})',color='grey')
        ax1.set_xlabel(f'CDD{cdd_threshold} in {year-cdd_mean_period+1}-{year} (°C.days)')
        ax1.legend()
        sns.scatterplot(data,x=f'gdp_per_capita_{year}',y=var,ax=ax2,legend=False,edgecolors='k',color='lightgrey')
        ax2.plot(data[f'gdp_per_capita_{year}'].values,Y_hat,ls='',marker='+',color='grey')
        ax2.set_xlabel(f'GDP per capita in {year} (rupees)')
        ax1.set_ylabel('Number of cooler systems in equiped households (#)')
        plt.subplots_adjust(wspace=0.1,top=0.92)
        # fig.suptitle(title)
        plt.savefig(os.path.join('output','cooler_equipment',f'cdd{cdd_threshold}_{save}_scatter.png'), bbox_inches='tight')
        plt.show()
    
    #%% Illustration des fonctions
    if False:
        illustrate_functions(a=0.9,b=1/300,c=1000,mode='Sigmoid',X_max=2500)
        illustrate_functions(a=0.9,b=1/43000,c=120000,mode='Sigmoid',X_max=350000)
        # illustrate_functions(b=0.9,c=3,mode='Exponential')
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()