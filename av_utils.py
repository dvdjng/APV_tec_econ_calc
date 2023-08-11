# -*- coding: utf-8 -*-
"""
Created on May 08, 2023

@author: David Jung
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import os
import pandas as pd
import numpy as np
import pytz
import datetime

import requests
import pvlib
from pvlib.location import Location
from pvlib import tracking
from pvlib.bifacial.pvfactors import pvfactors_timeseries
from pvlib import temperature
from pvlib import pvsystem

import pvfactors
from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray


sys.path.append('../lib/')


def tmy_download(latitude, longitude, tz):
    try: 
        # API request for points in df
        	
        data_pvgis  = pvlib.iotools.get_pvgis_tmy(latitude, longitude, outputformat='json', usehorizon=True, userhorizon=None, startyear=None, endyear=None, url='https://re.jrc.ec.europa.eu/api/v5_2/', map_variables=None, timeout=30)
        # data_pvgis  = pvlib.iotools.get_pvgis_tmy(-33.8991, -70.732, outputformat='json', usehorizon=True, userhorizon=None, startyear=None, endyear=None, url='https://re.jrc.ec.europa.eu/api/v5_2/', map_variables=None, timeout=30)
        # Get altitude and add to df
        altitude = data_pvgis[2].get("location").get("elevation")

        # Get tmy data 
        tmy_pvg_r = data_pvgis[0]

        # Check PVGIS Download
        if (tmy_pvg_r["G(h)"].sum() < 1):
            PVGIS_dl = "missing solar data"
        elif (tmy_pvg_r["WS10m"].sum() < 1) | (tmy_pvg_r["T2m"].sum() < 1):
            PVGIS_dl = "missing climate data"
        else:
            PVGIS_dl = "complete"


        # process tmy data: move 3 first rows to back to convert to Chilean time 
        timezone = pytz.timezone(tz)
        dt = datetime.datetime.utcnow()
        offset_seconds = timezone.utcoffset(dt).seconds
        offset_hours = offset_seconds / 3600.0
        if offset_hours > 12:
            offset_hours = offset_hours - 24
        

        if offset_hours < 0:
            offset_hours = -offset_hours
            offset_hours = int(offset_hours)
            # Extract the first three rows
            first_rows = tmy_pvg_r.head(offset_hours)

            # Remove the first three rows from the original DataFrame
            tmy_pvg_r  = tmy_pvg_r.drop(tmy_pvg_r.index[:offset_hours])

            # Concatenate the modified DataFrame and the first three rows
            tmy = pd.concat([tmy_pvg_r, first_rows])

            tmy["time"] = pd.date_range(start = "2022-01-01 00:00", end="2022-12-31 23:00", freq="h", tz=tz)
            
        
        else:
            print("TMY download only for South America")

                      
        # process tmy data: Rename for pvlib
        cols_to_use = [ "time", "T2m", "G(h)", "Gb(n)", "Gd(h)", "IR(h)", "WS10m", "RH", "SP"] 
        pvlib_column_names = ["time", "temp_air", "ghi", "dni", "dhi", "lwr_u", "wind_speed", "rh", "sp" ] #"info", "info_values",
        tmy = tmy[cols_to_use]
        tmy.columns = pvlib_column_names

        location = Location(latitude, longitude, tz, altitude)
        # Get solar azimuth and zenith to store in tmy
        solar_position = location.get_solarposition(times=tmy.index)
        tmy["azimuth"] = solar_position["azimuth"] 
        tmy["zenith"] = solar_position["zenith"] 
        tmy["apparent_zenith"] = solar_position["apparent_zenith"] 
        tmy = tmy.reset_index(drop=True)
            
    except requests.HTTPError as err:
        PVGIS_dl = err

    print("Download of TMY data: "+ str(PVGIS_dl))
    return tmy, altitude


def pv_yield(tmy_data, albedo, track, pvrow_azimuth, pvrow_tilt, n_pvrows, pvrow_width, pvrow_pitch, pvrow_height, bifaciality, observed_row): # pvrow_tilt with tracking == True is equal to max tilt
    
    #Definition of PV array
    gcr = pvrow_width / pvrow_pitch
    axis_azimuth = pvrow_azimuth - 90
    
    pvarray_parameters = {
        'n_pvrows': n_pvrows,
        'axis_azimuth': axis_azimuth,
        'pvrow_height': pvrow_height,
        'pvrow_width': pvrow_width,
        'gcr': gcr
    }

    # Create an ordered PV array
    #pvarray = OrderedPVArray.init_from_dict(pvarray_parameters) # ground is not initalized: https://github.com/SunPower/pvfactors/blob/master/pvfactors/geometry/pvarray.py#L12

    if track == True:
        # pv-tracking algorithm to get pv-tilt
        orientation = tracking.singleaxis(tmy_data['apparent_zenith'],
                                        tmy_data['azimuth'],
                                        max_angle=pvrow_tilt,
                                        backtrack=True,
                                        gcr=gcr
                                        )
        
        tmy_data['surface_azimuth'] = orientation['surface_azimuth']
        tmy_data['surface_tilt'] = orientation['surface_tilt'] 
    else:
        tmy_data['surface_azimuth'] = np.where((tmy_data["apparent_zenith"] > 0 ) & (tmy_data["apparent_zenith"] < 90), pvrow_azimuth,np.nan)
        tmy_data['surface_tilt'] = np.where((tmy_data["apparent_zenith"] > 0 ) & (tmy_data["apparent_zenith"] < 90), pvrow_tilt,np.nan)
    
    
    irrad = pvfactors_timeseries(tmy_data['azimuth'],
                             tmy_data['apparent_zenith'],
                             tmy_data['surface_azimuth'],
                             tmy_data['surface_tilt'],
                             axis_azimuth,
                             tmy_data.index,
                             tmy_data['dni'],
                             tmy_data['dhi'],
                             gcr,
                             pvrow_height,
                             pvrow_width,
                             albedo,
                             n_pvrows=n_pvrows,
                             index_observed_pvrow=observed_row
                             )

    # turn into pandas DataFrame
    irrad = pd.concat(irrad, axis=1)

    # using bifaciality factor and pvfactors results, create effective irradiance
    effective_irrad_bifi = irrad['total_abs_front'] + (irrad['total_abs_back']
                                                    * bifaciality)

    # get cell temperature using the Faiman model    - Here heat coefficients could be implemented
    temp_cell = temperature.faiman(effective_irrad_bifi, temp_air=25,
                                wind_speed=1)

    # using the pvwatts_dc model and parameters detailed above,
    # set pdc0 and return DC power for both bifacial and monofacial
    pdc0 = 1000
    gamma_pdc = -0.0043
    pdc_bifi = pvsystem.pvwatts_dc(effective_irrad_bifi,
                                temp_cell,
                                pdc0,
                                gamma_pdc=gamma_pdc
                                ).fillna(0)
    
    pac0 = 1000
    results_ac = pvlib.inverter.pvwatts(
            pdc=pdc_bifi, 
            pdc0=pac0, 
            eta_inv_nom=0.961,
            eta_inv_ref=0.9637)

    losses = pvlib.pvsystem.pvwatts_losses(
            soiling=5, 
            shading=3, 
            snow=0, 
            mismatch=2, 
            wiring=2, 
            connections=0.5, 
            lid=1.5, 
            nameplate_rating=1, 
            age=0, 
            availability=3)

    results_ac_real = results_ac * (1-losses/100)

    return results_ac_real


def shadow_ratio(tmy_data, albedo, track, pvrow_azimuth, pvrow_tilt, n_pvrows, pvrow_width, pvrow_pitch, pvrow_height): # pvrow_tilt with tracking == True is equal to max tilt
    
    #Definition of PV array
    gcr = pvrow_width / pvrow_pitch
    axis_azimuth = pvrow_azimuth + 90
    
    pvarray_parameters = {
        'n_pvrows': n_pvrows,
        'axis_azimuth': axis_azimuth,
        'pvrow_height': pvrow_height,
        'pvrow_width': pvrow_width,
        'gcr': gcr
    }

    # Create an ordered PV array
    pvarray = OrderedPVArray.init_from_dict(pvarray_parameters) # ground is not initalized: https://github.com/SunPower/pvfactors/blob/master/pvfactors/geometry/pvarray.py#L12

    if track == True:
        # pv-tracking algorithm to get pv-tilt
        orientation = tracking.singleaxis(tmy_data['apparent_zenith'],
                                        tmy_data['azimuth'],
                                        max_angle=pvrow_tilt,
                                        backtrack=True,
                                        gcr=gcr
                                        )
        tmy_data['surface_azimuth'] = orientation['surface_azimuth']
        tmy_data['surface_tilt'] = orientation['surface_tilt'] 
    else:
        tmy_data['surface_azimuth'] = np.where((tmy_data["apparent_zenith"] > 0 ) & (tmy_data["apparent_zenith"] < 90), pvrow_azimuth,np.nan)
        tmy_data['surface_tilt'] = np.where((tmy_data["apparent_zenith"] > 0 ) & (tmy_data["apparent_zenith"] < 90), pvrow_tilt,np.nan)
    
    # Create engine using the PV array
    engine = PVEngine(pvarray) 

    # Fit engine to data: which will update the pvarray object as well
    engine.fit(tmy_data.index, tmy_data.dni, tmy_data.dhi,
            tmy_data.zenith, tmy_data.azimuth,
            tmy_data.surface_tilt, tmy_data.surface_azimuth,
            albedo= albedo)

    """"
    a = pd.DataFrame()
    for i in range(0,len(pvarray.ts_ground.all_ts_surfaces)):
        a[str(i)+"_0,0"] = pvarray.ts_ground.all_ts_surfaces[i].coords.as_array[0][0]
        a[str(i)+"_1,0"] = pvarray.ts_ground.all_ts_surfaces[i].coords.as_array[1][0]

    # set x_min and x_max so that only area under PV array is considered (between second and second to last row)
    a[a < pvrow_pitch] = pvrow_pitch
    a[a > pvrow_pitch * (n_pvrows -2)] = pvrow_pitch * (n_pvrows -2)

    # sum up the shadow and ilum lenghts
    for i in range(0,len(pvarray.ts_ground.all_ts_surfaces)):
        a[str(i)] = a[str(i)+"_1,0"] - a[str(i)+"_0,0"]

    print("shadow ratio is calculated between x = "+str(pvrow_pitch)+" m and "+str(pvrow_pitch * (n_pvrows -2))+" m")

    shadow = 0
    for i in range(0,pvarray.ts_ground.n_ts_shaded_surfaces):
        shadow += a[str(i)]

    light = 0
    for i in range(pvarray.ts_ground.n_ts_shaded_surfaces,len(pvarray.ts_ground.all_ts_surfaces)):
        light += a[str(i)]


    sl = pd.DataFrame()
    sl["lenght_shadow"] = shadow
    sl["lenght_ilum"] = light
    sl["sum"] = sl["lenght_ilum"]+sl["lenght_shadow"]
    sl["shadow_ratio"] = 1 / sl["sum"] * sl["lenght_shadow"]

    sl, 
    """

    return pvarray

