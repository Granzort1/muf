ob1_min
ob2_min
ob3_min
ob4_min
ob5_min
ob6_min
ob7_min
ob8_min
ob9_min
ob10_min
self_academy
self_daycare
self_inparking
self_insports
self_under_station

if pollutant == 'pm10':
    model_name_4th = "PM10"
    model_name_label = "ug/m³"
    cut_conc = 200
    col = 'red'
    scale_y = "symlog"
    scale_name = "log"
elif pollutant == 'pm25':
    model_name_4th = "PM2.5"
    model_name_label = "ug/m³"
    cut_conc = 200
    col = 'red'
    scale_y = "symlog"
    scale_name = "log"
elif pollutant == 'pm1':
    model_name_4th = "PM1.0"
    model_name_label = "ug/m³"
    cut_conc = 200
    col = 'red'
    scale_y = "symlog"
    scale_name = "log"
elif pollutant == 'humi':
    model_name_4th = "Relative Humidity"
    model_name_label = "%"
    cut_conc = 100
    col = 'None'
    scale_y = "linear"
    scale_name = "linear"
elif pollutant == 'temp':
    model_name_4th = "Temperature"
    model_name_label = "°C"
    cut_conc = 10
    col = 'None'
    scale_y = "linear"
    scale_name = "linear"
elif pollutant == 'hcho':
    model_name_4th = "HCHO"
    model_name_label = "ug/m³"
    cut_conc = 100
    col = 'red'
    scale_y = "symlog"
    scale_name = "log"
elif pollutant == 'co':
    model_name_4th = "CO"
    model_name_label = "ppm"
    cut_conc = 25
    col = 'red'
    scale_y = "symlog"
    scale_name = "log"
elif pollutant == 'no2':
    model_name_4th = "NO2"
    model_name_label = "ppb"
    cut_conc = 300
    col = 'red'
    scale_y = "symlog"
    scale_name = "log"
elif pollutant == 'rn':
    model_name_4th = "Rn"
    model_name_label = "Bq/m³"
    cut_conc = 148
    col = 'red'
    scale_y = "symlog"
    scale_name = "log"
elif pollutant == 'voc':
    model_name_4th = "VOC"
    model_name_label = "ug/m³"
    cut_conc = 1000
    col = 'red'
    scale_y = "symlog"
    scale_name = "log"
elif pollutant == 'co2':
    model_name_4th = "CO2"
    model_name_label = "ppm"
    cut_conc = 1000
    col = 'red'
    scale_y = "symlog"
    scale_name = "log"
elif pollutant == 'tab':
    model_name_4th = "TAB"
    model_name_label = "CFU/m³"
    cut_conc = 100
    col = 'None'
    scale_y = "symlog"
    scale_name = "log"
else:
    pass