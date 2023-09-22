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
import pyet

import requests
import pvlib
from pvlib.location import Location
from pvlib import tracking
from pvlib.bifacial.pvfactors import pvfactors_timeseries
from pvlib import temperature
from pvlib import pvsystem
import numpy_financial as npf
import pvfactors
from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def daily_climate_data(df):
    df.index = df['time'] 
    df = df.fillna(method="ffill")
    df_d = pd.DataFrame()

    # inputs for pyet function
    if "ghi" in df.columns:
        df_d["rs_kW_m2"] = df["ghi"].resample("D").sum() / 1000
        df_d["rs_MJ_m2"] = df_d["rs_kW_m2"] *3.6

    if "temp_air" in df.columns:
        df_d["tmean_C"] = df["temp_air"].resample("D").mean()
        df_d["tmax_C"] = df["temp_air"].resample("D").max()
        df_d["tmin_C"] = df["temp_air"].resample("D").min()

    if "rh" in df.columns:
        df_d["rh_%"] = df["rh"].resample("D").mean()
        df_d["rhmax_%"] = df["rh"].resample("D").max()
        df_d["rhmin_%"] = df["rh"].resample("D").min()

    if "wind_speed" in df.columns:
        df_d["wind_speed_m_s"] = df["wind_speed"].resample("D").mean()
        df_d["wind_speedmax_m_s"] = df["wind_speed"].resample("D").max()

    if "rain" in df.columns:
        df_d["rain_mm"] = df["rain"].resample("D").sum()
    return df_d

def plot_climate_data(df_d,selected_data1 = 'rs_kW_m2', selected_data2 = 'wind_speed_m_s'):
    # Plot Irradation values

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_d.index, y=df_d[selected_data2], name=selected_data2), secondary_y=True)
    fig.add_trace(go.Scatter(x=df_d.index, y=df_d[selected_data1], name=selected_data1))



    # Edit the layout
    fig.update_layout(title='Datos Estacion Mocambinho INMET 2022 - Datos diarios: Irradiacíon Solar y Velocidad de Viento', 
        width=1000,  # Set the width to 600 pixels
        height=400)  # Set the height to 400 pixels) 
    fig.update_yaxes(title_text="Velocidad de viento (m/s)", secondary_y=True, range=[0, 5])
    fig.update_yaxes(title_text="Irradiación Solar (kWh/dia)", secondary_y=False, range=[0, 10])
    
    fig.show()    

def calc_evapo(df_d,lat,alt,method1="FAO-56",method2="Hargreaves"):

    df_pet = pyet.calculate_all(
        tmean=df_d["tmean_C"], 
        wind=df_d["wind_speed_m_s"], 
        rs=df_d["rs_MJ_m2"], 
        elevation=alt, 
        lat=pyet.deg_to_rad(lat), 
        tmax=df_d["tmean_C"], 
        tmin=df_d["tmin_C"], 
        rh=df_d["rh_%"], 
        rhmin=df_d["rhmin_%"], 
        rhmax=df_d["rhmax_%"])
    print("The values for total anual Evapotranspiration [mm/m2]are")
    print(round(df_pet.sum(),2))

    df_pet = df_pet[[method1,method2]]
    
    fig = go.Figure()
    for column in df_pet.columns:
        trace = go.Scatter(x=df_pet.index, y=df_pet[column], mode='lines', name=column)
        fig.add_trace(trace)
        # Edit the layout
        fig.update_layout(title='Computación de evpotranspiración', yaxis_title='Evapotranspiración diaria (mm/m2)') # 
        
    fig.show()
    return df_pet

def climate_summary(df_d, df_pet):
    climate_results = pd.DataFrame()
    for var in range(0,len(df_d.columns.values)):
        climate_results.loc[var,"parameter"] = df_d.columns.values[var]
        climate_results.loc[var,"yearly total"] = df_d[df_d.columns.values[var]].sum()
        climate_results.loc[var,"daily average"] = df_d[df_d.columns.values[var]].mean()
        climate_results.loc[var,"max month"] = df_d.groupby(df_d.index.month)[df_d.columns.values[var]].mean().idxmax()
        climate_results.loc[var,"max month value"] = df_d.groupby(df_d.index.month)[df_d.columns.values[var]].mean().max()
        climate_results.loc[var,"min month"] = df_d.groupby(df_d.index.month)[df_d.columns.values[var]].mean().idxmin()
        climate_results.loc[var,"min month value"] = df_d.groupby(df_d.index.month)[df_d.columns.values[var]].mean().min()
        i = var

    for var in range(0,len(df_pet.columns.values)):
        climate_results.loc[var+i+1,"parameter"] = df_pet.columns.values[var]
        climate_results.loc[var+i+1,"yearly total"] = df_pet[df_pet.columns.values[var]].sum()
        climate_results.loc[var+i+1,"daily average"] = df_pet[df_pet.columns.values[var]].mean()
        climate_results.loc[var+i+1,"max month"] = df_pet.groupby(df_pet.index.month)[df_pet.columns.values[var]].mean().idxmax()
        climate_results.loc[var+i+1,"max month value"] = df_pet.groupby(df_pet.index.month)[df_pet.columns.values[var]].mean().max()
        climate_results.loc[var+i+1,"min month"] = df_pet.groupby(df_pet.index.month)[df_pet.columns.values[var]].mean().idxmin()
        climate_results.loc[var+i+1,"min month value"] = df_pet.groupby(df_pet.index.month)[df_pet.columns.values[var]].mean().min()

    climate_results["max month value"] = climate_results["max month value"].round(2)
    climate_results["min month value"] = climate_results["min month value"].round(2)


    climate_results['max month'] = climate_results['max month'].astype(int)
    climate_results['min month'] = climate_results['min month'].astype(int)

    numbers_to_months = {
        1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
    }

    climate_results['max month'] = climate_results['max month'].map(numbers_to_months)
    climate_results['min month'] = climate_results['min month'].map(numbers_to_months) 



    climate_results["yearly total"] = climate_results["yearly total"].round(2)
    climate_results["daily average"] = climate_results["daily average"].round(2)
    climate_results.index=climate_results.parameter
    climate_results = climate_results.drop(columns=["parameter"])


    print(climate_results)

def pv_yield(tmy_data, albedo, track, pvrow_azimuth, pvrow_tilt, n_pvrows, pvrow_width, pvrow_pitch, pvrow_height, bifaciality): # pvrow_tilt with tracking == True is equal to max tilt
    
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
                             index_observed_pvrow=2
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
    pv_sum = results_ac_real.sum()
    return pv_sum, results_ac_real
    #return results_ac
#con pvgen = generacion especifica en un año en kWh/kWp/a

def lcoe_calc(pv_gen, kWp, capex,  wacc ,opex, degre = 0.005, inflation = 0.03, N = 25):

    # LCOE calculation (can be externalized as functions in av_utils)

    cashflow= pd.DataFrame(index=range(0,N))

    cashflow["OPEX_des"] = (opex * kWp * (1+inflation)**cashflow.index) / (1+wacc)**cashflow.index

    cashflow["EG_des"] = (pv_gen * kWp * (1-degre)**cashflow.index) / (1+wacc)**cashflow.index

    LCOE =  ((capex * kWp + cashflow["OPEX_des"].sum() ) / cashflow["EG_des"].sum())*1000
    
    

    print("LCOE of the simulated system is "+str(round(LCOE,2))+" USD/kWh")

    return LCOE

def tir(opex,capex,pv_gen,kWp,price,inflation=0.03,N=25,degre=0.005):
    
    cashflow= pd.DataFrame(index=range(0,N))

    flujos =[]
    
    gasto_inicial = -capex * kWp

    flujos.append(gasto_inicial)
        
    for i in range(1,N):

        flujo_generado = (pv_gen * kWp * (1-degre)**i)*price
        flujo_negativo = -(opex * kWp * (1+inflation)**i) 
        neto = flujo_generado+flujo_negativo
        flujos.append(neto)
    
    cashflow["Flujos"] = flujos

    TIR = (npf.irr(flujos))
    return TIR,cashflow
