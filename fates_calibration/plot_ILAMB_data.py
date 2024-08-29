import os
import xarray as xr
import matplotlib.pyplot as plt
from plotting_functions import plot_obs_bylat, plot_global, get_by_lat

def main(file, out_dir):
    
    obs_dict = {
      'ALBEDO':{
        'models': ['CERESed4.1', 'GEWEX.SRB'],
        'var': 'albedo',
        'name': 'albedo',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': '0-1',
        'lat_units': '0-1',
        'cmap': 'jet',
        'div': False,
        },
      'BIOMASS':{
        'models': ['ESACCI', 'GEOCARBON'],
        'var': 'biomass',
        'name': 'forest biomass',
        'convert_units': 'Tg',
        'conversion_factor': 0.01*1e-6,
        'global_units': 'MgC ha$^{-1}$',
        'lat_units': 'TgC',
        'cmap': 'YlGn',
        'div': False,
        },
      'BURNTAREA':{
        'models': ['GFED4.1S'],
        'var': 'burntarea',
        'name': 'burned area',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': '%',
        'lat_units': '%',
        'cmap': 'Reds',
        'div': False,
      },
      'EF':{
        'models': ['FLUXCOM', 'CLASS', 'WECANN', 'GBAF'],
        'var': 'ef',
        'name': 'evaporative fraction',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': '0-1',
        'lat_units': '0-1',
        'cmap': 'BrBG',
        'div': False,
      },
      'FIRE':{
        'models': ['CERESed4.1', 'GEWEX.SRB'],
        'var': 'fire',
        'name': 'surface upward longwave radiation flux',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'W m$^{-2}$',
        'lat_units': 'W m$^{-2}$',
        'cmap': 'jet',
        'div': False,
    },
    'FSA':{
        'models': ['CERESed4.1', 'GEWEX.SRB'],
        'var': 'fsa',
        'name': 'surface net shortwave radiation flux',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'W m$^{-2}$',
        'lat_units': 'W m$^{-2}$',
        'cmap': 'jet',
        'div': False,
    },
    'FSR':{
        'models': ['CERESed4.1', 'GEWEX.SRB'],
        'var': 'fsr',
        'name': 'surface upward shortwave radiation flux',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'W m$^{-2}$',
        'lat_units': 'W m$^{-2}$',
        'cmap': 'jet',
        'div': False,
    },
    'GPP':{
        'models': ['FLUXCOM', 'WECANN', 'GBAF'],
        'var': 'gpp',
        'name': 'GPP',
        'convert_units': 'PgC yr-1',
        'conversion_factor': 1e6*1e-12,
        'global_units': 'kgC m$^{-2}$ yr$^{-1}$',
        'lat_units': 'PgC yr$^{-1}$',
        'cmap': 'YlGn',
        'div': False,
    },
    'GR':{
        'models': ['CLASS'],
        'var': 'gr',
        'name': 'ground heat flux',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'W m$^{-2}$',
        'lat_units': 'W m$^{-2}$',
        'cmap': 'Oranges',
        'div': False,
    },
    'LAI':{
        'models': ['AVHRR', 'AVH15C1'],
        'var': 'lai',
        'name': 'LAI',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'm$^2$ m$^{-2}$',
        'lat_units': 'm$^2$ m$^{-2}$',
        'cmap': 'Oranges',
        'div': False,
    },
    'LE':{
        'models': ['FLUXCOM', 'DOLCE', 'CLASS', 'WECANN', 'GBAF'],
        'var': 'le',
        'name': 'latent heat flux',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'W m$^{-2}$',
        'lat_units': 'W m$^{-2}$',
        'cmap': 'YlGnBu',
        'div': False,
    },
    'MRRO':{
        'models': ['LORA', 'CLASS'],
        'var': 'mrro',
        'name': 'runoff',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'mm day$^{-1}$',
        'lat_units': 'mm day$^{-1}$',
        'cmap': 'Blues',
        'div': False,
    },
    'NEE':{
        'models': ['FLUXCOM'],
        'var': 'nee',
        'name': 'NEE',
        'convert_units': 'PgC yr-1',
        'conversion_factor': 1e6*1e-12,
        'global_units': 'kgC m$^{-2}$ yr$^{-1}$',
        'lat_units': 'PgC yr$^{-1}$',
        'cmap': 'RdYlGn_r',
        'div': True,
    },
    'RLNS':{
        'models': ['CERESed4.1', 'GEWEX.SRB'],
        'var': 'rlns',
        'name': 'surface net longwave radiation flux',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'W m$^{-2}$',
        'lat_units': 'W m$^{-2}$',
        'cmap': 'jet',
        'div': False,
    },
    'RN':{
        'models':  ['CERESed4.1', 'GEWEX.SRB', 'CLASS'],
        'var': 'rn',
        'name': 'net radiation',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'W m$^{-2}$',
        'lat_units': 'W m$^{-2}$',
        'cmap': 'jet',
        'div': False,
    },
    'SH':{
        'models':  ['FLUXCOM', 'CLASS', 'WECANN', 'GBAF'],
        'var': 'sh',
        'name': 'sensible heat flux',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'W m$^{-2}$',
        'lat_units': 'W m$^{-2}$',
        'cmap': 'jet',
        'div': False,
    },
    'SW':{
        'models':  ['WangMao'],
        'var': 'sw',
        'name': 'soil water',
        'convert_units': '',
        'conversion_factor': None,
        'global_units': 'kg m$^{-2}$',
        'lat_units': 'kg m$^{-2}$',
        'cmap': 'Blues',
        'div': False,
    },
    }
    global_dat = xr.open_dataset(file)
  
    for var, attributes in obs_dict.items():
        
        # get observations by latitude and convert global data if needed
        var_glob, var_bylat = get_by_lat(global_dat, attributes['var'], attributes['models'],
                                         attributes['convert_units'], attributes['conversion_factor'])
        
        # plot observations by latitude
        plot_obs_bylat(var_bylat, attributes['var'], attributes['name'], attributes['lat_units'])
        plt.savefig(f'{out_dir}/{var}_by_latitude.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # plot global observations
        plot_global(var_glob, attributes['name'], attributes['global_units'], attributes['cmap'],
                    div=attributes['div'])
        plt.savefig(f'{out_dir}/{var}_global.png', bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    out_dir = "/glade/u/home/afoster/FATES_Calibration/FATES_SP/observations/graphs"
    file = "/glade/u/home/afoster/FATES_Calibration/FATES_SP/observations/ILAMB_obs.nc"
    main(file, out_dir)
        
  
    