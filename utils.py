#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:57:23 2026

@author: amounier
"""

import time
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import pandas as pd
from datetime import date

def get_states():
    path = os.path.join('data','geometry','INDIA_STATES.geojson')
    data = gpd.read_file(path)
    data = data.set_crs(epsg=4326)
    data = data[['STCODE11', 'STNAME_SH', 'geometry']]
    data['STNAME_SH'] = data['STNAME_SH'].str.replace('&','and') 
    return data

states_geometries = get_states()

dict_code_name = states_geometries.set_index('STCODE11')['STNAME_SH'].to_dict()
dict_name_code = {v:k for k,v in dict_code_name.items()}
dict_code_geom = states_geometries.set_index('STCODE11')['geometry'].to_dict()
dict_name_geom = states_geometries.set_index('STNAME_SH')['geometry'].to_dict()

def save_figure(file_name):
    os.makedirs('output',exist_ok=True)
    
    folder,file = os.path.split(file_name)
    if bool(folder):
        os.makedirs(os.path.join('output',folder),exist_ok=True)
        
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    plt.savefig(os.path.join('output',folder,f'{today}_{file}.png'),bbox_inches='tight')
    return 

    
def get_extent():
    extent = [67.7, 97.7, 7.8, 35.1]
    return extent


def blank_map(dpi=300):
    fig = plt.figure(figsize=(7,7), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent(get_extent())
    
    ax.add_feature(cfeature.OCEAN, color='grey',zorder=2)
    ax.add_feature(cfeature.LAND, color='lightgrey',zorder=1)
    ax.add_feature(cfeature.COASTLINE,zorder=5)
    ax.add_feature(cfeature.BORDERS,zorder=3)
    return fig,ax


def draw_state_map(dict_vals,cbar_min=0,cbar_max=1.,
                   automatic_cbar_values=False, cbar_label=None, no_cbar=False,
                   map_title=None,save=None,cmap=None,zcl_label=False,alpha=None,
                   lw=0.5,figax=None,
                   border_color='k'):
    
    if figax is not None:
        fig,ax = figax
    else:
        fig,ax = blank_map()
    
    if cmap is None:
        cmap = matplotlib.colormaps.get_cmap('viridis')
    
    plotter = pd.DataFrame().from_dict({'climats':dict_vals.keys(),'vals':dict_vals.values()})
    
    if len(plotter.climats.values[0])<3:
        geom_dict = dict_code_geom
    else:
        geom_dict = dict_name_geom
        
    plotter['geometry'] = [geom_dict.get(d) for d in plotter.climats]
    plotter = gpd.GeoDataFrame(plotter, geometry=plotter.geometry)
    
    if automatic_cbar_values:
        cbar_max = plotter.vals.quantile(0.99)
        cbar_min = plotter.vals.quantile(0.01)
        cbar_extend = 'both'
    else:
        cbar_extend = 'neither'
    
    plotter['color'] = (plotter.vals-cbar_min)/(cbar_max-cbar_min)
    plotter['color'] = plotter['color'].apply(cmap)
    plotter['color'] = [(1.,1.,1.,1.) if e == (0.,0.,0.,0.) else e for e in plotter.color ]
    
    plotter.plot(color=plotter.color, ax=ax, transform=ccrs.PlateCarree(),alpha=alpha)
    plotter.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color=border_color,lw=lw)
        
    if not all(plotter.color==(1.,1.,1.,1.)) and not no_cbar:
        cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
        norm = matplotlib.colors.Normalize(vmin=cbar_min, vmax=cbar_max)
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        cbar_label_var = cbar_label
        _ = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend=cbar_extend, extendfrac=0.02)
        
    ax.set_title(map_title)
    if save is not None:
        save_figure(save)
        
    return fig,ax

#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # dict_states = dict_name_code.copy()
    dict_states = dict_code_name.copy()
    
    for k,v in dict_states.items():
        dict_states[k] = None
        
    draw_state_map(dict_states,save=os.path.join('maps','empty_state_map'))
    
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()