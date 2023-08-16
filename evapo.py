import streamlit as st
import pandas as pd
import numpy as np

import pyet

import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem

import av_utils as av

st.set_page_config(page_title="Evapotranspiracion", layout="wide")
st.subheader("Hola")


with st.container():
    st.subheader("Please select location ")
    st.title("test")
    st.write("test2")


#c = st.slider('FPV Capacity (kWp): ', 0,1000, 300, 50)
latitude = st.number_input("latitude", min_value=-50.0, max_value=-0.0, value=-15.087836)    
longitude = st.number_input("longitude", min_value=-80.0, max_value=-20.0, value=-44.015762)    
altitude = 450
tz= 'Brazil/East'  
#creating a sample data consisting different points 
df_pos = pd.DataFrame([[latitude, longitude]],columns=['latitude', 'longitude'])
st.map(df_pos)


# Dowload of TMY from PVGIS, altitude 
if st.button("Download TMY Data"):
    tmy, altitude = av.tmy_download(latitude, longitude, tz)
    st.write('successfully downloaded tmy data')
    tmy.to_csv('tmy.csv')
    st.line_chart(tmy.ghi)


uploaded_file = st.file_uploader("Choose a .csv file downloaded from https://bdmep.inmet.gov.br/#")
if uploaded_file is not None:
    # Read first data-table with data from october 21
    df = pd.read_csv(uploaded_file, sep=';',header=9)  #, index_col='TIMESTAMP', skiprows=[2,3], low_memory=False

    # Apply the function and create a new column
    df["hour"] = df["Hora Medicao"] /100

    # Combine date and hour columns into a single datetime column
    df['datetime'] = pd.to_datetime(df['Data Medicao']) + pd.to_timedelta(df['hour'], unit='h')
    df['datetime'] = df['datetime'].dt.tz_localize('GMT').dt.tz_convert(tz)
    # Set the datetime column as the index
    df.set_index('datetime', inplace=True)
    df.drop(columns=["Data Medicao", "Hora Medicao", "hour", "Unnamed: 22"], inplace=True)

    pvlib_column_names = ['rain',
        "sp",
        'PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB)',
        'PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT)(mB)',
        'PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT)(mB)',
        "ghi",'TEMPERATURA DA CPU DA ESTACAO(°C)',
        "temp_air",
        'TEMPERATURA DO PONTO DE ORVALHO(°C)',
        'temp_air_max',
        'temp_air_min',
        'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT)(°C)',
        'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT)(°C)',
        'TENSAO DA BATERIA DA ESTACAO(V)',
        'rh_max',
        'rh_min',
        "rh",
        'VENTO, DIRECAO HORARIA (gr)(° (gr))', 'VENTO, RAJADA MAXIMA(m/s)',
        "wind_speed"]

    df.columns = pvlib_column_names
    cols_to_use = ["temp_air",'temp_air_max','temp_air_min', "ghi","wind_speed", "rh",'rh_max', 'rh_min', "sp", "rain"]
    df= df[cols_to_use]
    df["ghi"] = np.where(df["ghi"] < 0 , 0, df["ghi"])
    df["ghi_W_m2"] = df["ghi"] /0.2777
    st.write(df)

if st.button("Calculate Evapotranspiration"):
    df = df.loc["2022-01-01 00:00:00-03:00":"2022-12-31 23:00:00-03:00"]
    print("The dataframe contains " + str(df.isnull().sum().sum())+ " NaN values")
    print(df[df.isna().any(axis=1)])
    df = df.fillna(method="ffill")

    # inputs for pyet function
    lati =  pyet.check_lat(pyet.deg_to_rad(latitude))

    tmax = df["temp_air_max"].resample("D").max()
    tmin = df["temp_air_min"].resample("D").min()
    tmean = ( tmax + tmin ) / 2

    rhmax = df["rh"].resample("D").max()
    rhmin = df["rh_max"].resample("D").min()
    rh = df["rh_min"].resample("D").mean()

    wind = df["wind_speed"].resample("D").mean()

    rs = df["ghi"].resample("D").sum() / 1000
    #rs_apv = tmy["ghi_apv"].resample("D").sum() * 0.0036

    # execuation of pyet function
    ev_pm_fao56 = pyet.pm_fao56(tmean, wind=wind, rs=rs, tmax=tmax, tmin=tmin, rh=rh, rhmin=rhmin, rhmax=rhmax, elevation=altitude, lat=lati)
    #ev_pm_fao56_apv = pyet.pm_fao56(tmean, wind=wind, rs=rs_apv, tmax=tmax, tmin=tmin, rh=rh, rhmin=rhmin, rhmax=rhmax, elevation=alt, lat=lati)

    #store resyults in tmy
    df["ET_mm_Penman-Monteith"] = df.index.map(ev_pm_fao56)
    #df["ET_apv_mm"] = df.index.map(ev_pm_fao56_apv)

    #print and visualize resutls
    print("yearly evapotranspiration for open field is "+str(round(df["ET_mm_Penman-Monteith"].sum(),2))+" l/m2")
    #print("yearly evapotranspiration under AV shading of "+str(round((tmy["shadow_ratio"] *tmy["ghi"]).sum() /tmy["ghi"].sum(),2)*100)+" % is "+str(round(tmy.ET_apv_mm.sum(),2))+" l/m2")

    st.line_chart(ev_pm_fao56)
    


tilt = st.slider('PV Surface tilt (deg): ', 0, 90, 10, 5)   
azimuth = st.slider('Azimut (deg): ', 0, 360, 0, 5)

# Inputs for PV simulation
track = True
pvrow_azimuth = azimuth
pvrow_tilt = tilt

# fijos
albedo = 0.2
n_pvrows = 3
pvrow_width = 1
pvrow_pitch = 6
pvrow_height = 1.5
bifaciality = 0.9

if st.button("Calculate PV Generation"):
    # PV simulation (pvlib viewfactors)
    tmy = pd.read_csv("tmy.csv", index_col=0)
    pv = av.pv_yield(tmy_data = tmy, 
                    albedo = albedo, 
                    track = track, 
                    pvrow_azimuth = pvrow_azimuth, 
                    pvrow_tilt = pvrow_tilt, 
                    n_pvrows = n_pvrows, 
                    pvrow_width = pvrow_width, 
                    pvrow_pitch = pvrow_pitch, 
                    pvrow_height = pvrow_height, 
                    bifaciality = bifaciality,
                    observed_row = 2)
    #print("PV generation is "+str(pv.sum()/1000)+" kWh/kWp/year")



    chart_data = pv
    st.line_chart(chart_data)
    st.write("The total FPV Generation is " + str(round(chart_data.sum()/1000,2))+ " kWh in one year")

#alb = st.slider('albedo %: :', 0,100, 8, 1)