"""Holds constants for FATES calibration workflow
"""

# mapping from FATES pft name to CLM PFT index
FATES_CLM_INDEX = {"not_vegetated": [0],
                "broadleaf_evergreen_tropical_tree": [4],
                "needleleaf_evergreen_extratrop_tree": [2, 1],
                "needleleaf_colddecid_extratrop_tree": [3],
                "broadleaf_evergreen_extratrop_tree": [5],
                "broadleaf_hydrodecid_tropical_tree": [6],
                "broadleaf_colddecid_extratrop_tree": [8, 7],
                "broadleaf_evergreen_extratrop_shrub": [9],
                "broadleaf_hydrodecid_extratrop_shrub": [11, 10],
                "broadleaf_colddecid_extratrop_shrub": [11, 10],
                "arctic_c3_grass": [12],
                "cool_c3_grass": [13],
                "c4_grass": [14],
                "c3_crop": [15],
                "c3_irrigated": [16]}

FATES_INDEX = {"broadleaf_evergreen_tropical_tree": 1,
                "needleleaf_evergreen_extratrop_tree": 2,
                "needleleaf_colddecid_extratrop_tree": 3,
                "broadleaf_evergreen_extratrop_tree": 4,
                "broadleaf_hydrodecid_tropical_tree": 5,
                "broadleaf_colddecid_extratrop_tree": 6,
                "broadleaf_evergreen_extratrop_shrub": 7,
                "broadleaf_hydrodecid_extratrop_shrub": 8,
                "broadleaf_colddecid_extratrop_shrub": 9,
                "arctic_c3_grass": 10,
                "cool_c3_grass": 11,
                "c4_grass": 12,
                "c3_crop": 13,
                "c3_irrigated": 14}

# FATES pft names and their IDs
FATES_PFT_IDS = {"broadleaf_evergreen_tropical_tree": "BETT",
            "needleleaf_evergreen_extratrop_tree": "NEET",
            "needleleaf_colddecid_extratrop_tree": "NCET",
            "broadleaf_evergreen_extratrop_tree": "BEET",
            "broadleaf_hydrodecid_tropical_tree": "BDTT",
            "broadleaf_colddecid_extratrop_tree": "BDET",
            "broadleaf_evergreen_extratrop_shrub": "BEES",
            "broadleaf_colddecid_extratrop_shrub": "BCES",
            "arctic_c3_grass": "AC3G",
            "cool_c3_grass": "C3G",
            "c4_grass": "C4G",
            "c3_crop": "C3C",
            "c3_irrigated": "C3CI"}

# units for variables
VAR_UNITS = {'GPP': 'kgC m$^{-2}$ yr$^{-1}$',
        'EFLX_LH_TOT': 'W m$^{-2}$',
        'FSH': 'W m$^{-2}$',
        'EF': '',
        'SOILWATER_10CM': 'mm yr$^{-1}$',
        'ASA': '',
        'FSR': 'W m$^{-2}$',
        'FSA': 'W m$^{-2}$',
        'FIRE': 'W m$^{-2}$',
        'RLNS': 'W m$^{-2}$',
        'RN': 'W m$^{-2}$'}

# observation variable names for each modeled variable name
OBS_MODEL_VARS = {'GPP': 'gpp',
        'EFLX_LH_TOT': 'le',
        'FSH': 'sh',
        'EF': 'ef',
        'SOILWATER_10CM': 'sw',
        'ASA': 'albedo',
        'FSR': 'fsr',
        'FSA': 'fsa',
        'FIRE': 'fire',
        'RLNS': 'rlns',
        'RN': 'rn'}

ILAMB_MODELS = {
      'ALBEDO': ['CERESed4.1', 'GEWEX.SRB'],
      'BIOMASS': ['ESACCI', 'GEOCARBON'],
      'BURNTAREA': ['GFED4.1S'],
      'EF': ['FLUXCOM', 'CLASS', 'WECANN', 'GBAF'],
      'FIRE': ['CERESed4.1', 'GEWEX.SRB'],
      'FSA': ['CERESed4.1', 'GEWEX.SRB'],
      'FSR': ['CERESed4.1', 'GEWEX.SRB'],
      'GPP': ['FLUXCOM', 'WECANN', 'GBAF'],
      'GR': ['CLASS'],
      'LAI': ['AVHRR', 'AVH15C1'],
      'LE': ['FLUXCOM', 'DOLCE', 'CLASS', 'WECANN', 'GBAF'],
      'MRRO': ['LORA', 'CLASS'],
      'NEE': ['FLUXCOM'],
      'RLNS': ['CERESed4.1', 'GEWEX.SRB'],
      'RN': ['CERESed4.1', 'GEWEX.SRB', 'CLASS'],
      'SH': ['FLUXCOM', 'CLASS', 'WECANN', 'GBAF'],
      'SW': ['WangMao']
     }