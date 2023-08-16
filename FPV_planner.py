import pandas as pd
import streamlit as st
import evapo as ev

import pvlib
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

tmy = pd.read_csv(r'FPV_results.csv', index_col=0)
tmy.index = pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq="H", tz="America/Santiago")
module = pvlib.pvsystem.retrieve_sam('SandiaMod')["Canadian_Solar_CS5P_220M___2009_"]
inverter = pvlib.pvsystem.retrieve_sam('CECInverter')["iPower__SHO_1_1__120V_"]
temperature_parameters = {'u_c': 29.0, 'u_v': 0} #TEMPERATURE_MODEL_PARAMETERS['pvsyst']['freestanding']

altitude = 405

#c = st.slider('FPV Capacity (kWp): ', 0,1000, 300, 50)
lat = st.number_input("latitude", min_value=-50.0, max_value=-20.0, value=-33.8991)    
long = st.number_input("longitude", min_value=-80.0, max_value=-68.0, value=-70.7320)    
#creating a sample data consisting different points 

df_pos = pd.DataFrame([[lat, long]],columns=['latitude', 'longitude'])

#plotting a map with the above defined points

st.map(df_pos)



if st.button("Download TMY Data"):
    tmy_pvg_r  = pvlib.iotools.get_pvgis_tmy(lat, long, outputformat='json', usehorizon=True, userhorizon=None, startyear=None, endyear=None, url='https://re.jrc.ec.europa.eu/api/v5_2/', map_variables=None, timeout=30)[0]
    # move 3 first rows to back to convert to Chilean time 
    tmy_pvg = tmy_pvg_r.iloc[3:].append(tmy_pvg_r.iloc[:3])
    tmy_pvg.index = pd.date_range(start = "2022-01-01 00:00", end="2022-12-31 23:00", freq="h", tz="America/Santiago")
    # Rename for pvlib
    cols_to_use = ["T2m", "G(h)", "Gb(n)", "Gd(h)", "IR(h)", "WS10m", "RH", "SP"]
    pvlib_column_names = ["temp_air", "ghi", "dni", "dhi", "lwr_u", "wind_speed", "rh", "sp" ]
    tmy_pvg = tmy_pvg[cols_to_use]
    tmy_pvg.columns = pvlib_column_names
    tmy = tmy_pvg
    st.write('successfully updated tmy data')
else:
    st.write('Press to update tmy data')

tilt = st.slider('PV Surface tilt (deg): ', 0, 90, 10, 5)   
azimuth = st.slider('Azimut (deg): ', 0, 360, 0, 5)

system = PVSystem(surface_tilt = tilt, surface_azimuth=azimuth, module_parameters=module, inverter_parameters=inverter, temperature_model_parameters=temperature_parameters, modules_per_string=4, strings_per_inverter=1)
location = Location(lat, long, tz = 'America/Santiago', altitude = altitude)

m = ModelChain(system,location)
m.run_model(tmy)
chart_data = m.results.ac
st.line_chart(chart_data)
st.write("The total FPV Generation is " + str(round(chart_data.sum()/1000,2))+ " kWh in one year")

alb = st.slider('albedo %: :', 0,100, 8, 1)


chart_data2 = ev.calculate_pet(tmy["sp"], tmy["temp_air"], tmy["rh"], tmy["wind_speed"], tmy["ghi"], tmy["lwr_u"], alb/100, 0, "hourly")
#chart_data2.index = pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq="H", tz="America/Santiago")

st.line_chart(chart_data2)
st.write("The total evaporation is " + str(round(chart_data2.sum(),2))+ " mm/m2 in one year")