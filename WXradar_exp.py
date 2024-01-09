import os
import h5py
import argparse
import sys
from datetime import datetime, timedelta
import numpy as np
import math
import simplekml
from geopy import distance, location
import pandas as pd
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt


class WXradar_exp:


  def __init__(self):
    
    self.main_path = ""
    self.az_range = ['26.0', '27.0', '28.0']                                      # Angulos de AZ del experimento
    self.logs_path = ""                                                           # Ruta de los logs de vuelo
    self.drone_csv_path = "/home/gibs/SOPHY_CALIBRATION/results_exp/csv"          # Ruta de guardado de archivos .csv
    self.drone_hdf5_path = ""                                                     # Ruta de guardado de archivos .hdf5

    self.SOPHY_pos = (-11.9527, -76.875858, 522 + 2)                              # Posición de SOPHy durante el experimento
    self.cable_length = 30                                                        # Longitud de la cuerda
    self.sphere_rad  = 0.178                                                      # Radio de la esfera

    self.threshold = -45.2                                                        # Umbral de deteccion de potencia rx (dB) 
    self.sampling_rate = 10e6                                                     # Frecuencia de muestreo (MHz)
    self.IPP_km = 600                                                             # IPP - km
    self.IPP_t = 400e-6                                                           # IPP - s                      

    self.range_resolution = (self.IPP_km * self.sampling_rate) / self.IPP_t       # Resolucion en rango

    self.hdf5_path = ""                                                           # Ruta de acceso a la data procesada
    self.beam_width = math.radians(1.8)                                           # Ancho del beam de la antena
    self.pulse_width = 0.1e-6                                                     # Ancho del pulso durante el experimento
    self.antenna_gain_lnr = 10**(38.5/10)                                         # Ganancia de la antena (lineal)   
    self.freq_oper = 9.345e9                                                      # Frecuencia de operacion del radar

    pTdbm = 44                                                                    # Potencia de transmision (dBm)
    self.pT = 10**((pTdbm-30)/10)                                                 # Potencia medida a la salida del TX (W) 

    self.sigma = math.pi * self.sphere_rad ** 2                                   # Radar Cross Section de la esfera 
    self.gLNA = 70                                                                # Ganancia total del LNA (dB)
    self.k_M = 0.93                                                               # Coeficiente de refraccion del medio 

    #Variables particulares de c/ experimento

    self.idx_min_range_bin = 94                                                   # Indice del primer range bin de deteccion                              
    self.idx_t_conm = 87                                                          # Indice del range bin de t/conmutacion
    self.idx_file = 0
    self.idx_exp_ele = 20

    """
    self.telem = "2023-12-12 22-22-31.csv"
    self.range_offset = 0                                                              
    self.offset_hora = 0
    """
    self.linewidth_plot = 0

  def conv_logs(self):
    
    '''
    Lee un archivo .log, extrae los angulos roll, pitch, yaw junto a las coordenadas de GPS del drone y 
    la marca de tiempo en una entrada para luego en formar un dataset que se escribirá en un archivo hdf5, 
    csv y .kml.

    * Variable PATH_LOGS indica la ruta del archivo .log a leer
    * Los archivos procesados se guardan dentro del directorio principal en la carpeta /results_exp/ 

    '''

    def actualiza_gps(p):

      global myloc
      myloc["week_ms"] = float(p[3])
      myloc["week"] = float(p[4])
      myloc["lat"] = float(p[7])
      myloc["long"] = float(p[8])
      myloc["alt"] = float(p[9])

    def actualiza_fecha(p):

      global fecha_base
      global nueva_fecha
      global fecha
      global fecha_acord
      semanas = myloc["week"]
      micros = myloc["week_ms"]
      
      fecha_ref = fecha_base + timedelta(weeks = 100)
      
      if myloc["week"] == '0':
          fecha = fecha_base
      else:
          nueva_fecha = fecha_base + timedelta(weeks = semanas )
          nueva_fecha = nueva_fecha + timedelta(milliseconds = micros ) -timedelta(hours = 5) -timedelta(seconds= 18) #Peru -5 UTC
          if nueva_fecha > fecha :
              fecha = nueva_fecha
              
          if nueva_fecha > fecha_ref:
              fecha_acord = nueva_fecha
              
              
      print("FECHA: ",fecha)

    def sensor_data(p, sen,time):

        print(" ")
        #sensor = TimeUS,Año,Mes,dia,horas,minutos,segundos,micros,Lat,Lng,Alt,Roll,Pitch,Yaw
        #print("p1: ",time)
        sen[0] = float(p[1]) - time
        sen[1] = int(fecha.year)
        sen[2] = int(fecha.month)
        sen[3] = int(fecha.day)
        sen[4] = int(fecha.hour)
        sen[5] = int(fecha.minute)
        sen[6] = int(fecha.second)
        sen[7] = float(fecha.microsecond)
        sen[8] = myloc["lat"]
        sen[9] = myloc["long"]
        sen[10] = myloc["alt"]
        sen[11] = float(p[2])
        sen[12] = float(p[3])
        sen[13] = float(p[4])

        #for i in range(6):
        #    sen[11+i] = float(p[i+2])/1.0
        return sen

    def anade_data(sen, mat):
        mat = np.vstack((mat,sen))
        #print(" Dimension ")
        #print(mat.shape)
        return mat

    for log_file in sorted(os.listdir(self.logs_path)):

        kml = simplekml.Kml()
        time_zero = True
        global fecha, nueva_fecha, fecha_base, myloc, p, fecha_acord, t1
        t1 = 5.0
        time_us = 0
        myloc = {'week_ms': 0, 'week' : 0, 'lat' : 0, 'long' : 0, 'alt' : 0}
        fecha_base = datetime( 1980, month=1, day=6, hour=0, minute=0, second=0, microsecond=0)
        #sensor = TimeUS,Año,Mes,dia,horas,minutos,segundos,micros,Lat,Lng,Alt,Roll,Pitch,Yaw
        sensor = np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        matriz=np.zeros(shape=(1,14)) #matriz donde se almacenaran los datos extraidos de las distintas filas

        fecha = fecha_base
        #print(fecha_base)
        ruta = self.logs_path + "/" + log_file
        data = open(ruta).readlines()
        i=0
        fecha_inicio = fecha
        f_ini = False
        nlines=len(data)
        print(len(data))
        try:
           for line in data:
                p=line.replace(" ","")
                p=p.split(',')

                if time_zero:    
                    if p[0]=='IMU':         #se toma la pimera referencia de guardado, en multirrotores se puede
                        time_us=float(p[1]) #usar ACC1, en aviones usar IMU
                        time_zero = False

                if p[0]=='GPS':
                    actualiza_gps(p)
                    actualiza_fecha(p)
                    
                elif  p[0]=='ANG' :
                    if(not f_ini): #salva la primera fecha para el nombre del archivo
                         fecha_inicio = fecha
                         f_ini = True
                         #print("p[0]",p[0])
                    pnt = kml.newpoint(name=str(i), altitudemode ='absolute')
                    pnt.style.iconstyle.scale = 0.5
                    pnt.style.iconstyle.icon.href = 'http://maps.gstatic.com/mapfiles/ridefinder-images/mm_20_orange.png'
                    i = i+1
                    print(p)
                    sensor = sensor_data(p,sensor,time_us) #actualiza los datos a la lista
                    pnt.coords = [(sensor[9],sensor[8],sensor[10])]
                    descrp = "Roll: "+str(sensor[11])+"\t Pitch: "+ str(sensor[12]) +"\t Yaw: "+ str(sensor[13])
                    print(descrp)
                    pnt.description = descrp +'\n Date: '+fecha.strftime("%d-%m-%Y %H:%M:%S")

                    if not time_zero:
                        matriz = anade_data(sensor, matriz) #ingresa una nueva fila a la matriz de almacenaje

           fout = h5py.File(self.main_path[-4::]+"/results_exp/hdf5/"+fecha_acord.strftime("%Y%m%d")+fecha_acord.strftime("%H_%M_%S"),'w')
           kml.document.name = "copter"
           kml.save(self.main_path[-4::]+"/results_exp/kml/Copter_"+fecha_acord.strftime("%Y%m%d")+fecha_acord.strftime("%H_%M_%S")+".kml")
           dset = fout.create_dataset("dset", matriz.shape)
           #print(matriz.shape)
           dataset = fout['/dset']
           fout['/dset'].attrs.create('parametros',('TimeUS', 'Año', 'Mes', 'dia', 'horas', 'minutos', 'segundos', 'micros', 'Lat', 'Lng', 'Alt', 'Roll', 'Pitch', 'Yaw'),dtype=h5py.special_dtype(vlen=str))
           dataset[...] = matriz
           np.savetxt(self.main_path[-4::]+"/results_exp/csv/d"+fecha_acord.strftime("%Y%m%d")+fecha_acord.strftime("%H_%M_%S")+".csv", matriz,fmt='%10.7f', delimiter=',')
           fout.close() 


        finally:
            print("Finalizado...")

  def read_hdf5_drone(self):

    '''
    Lee un archivo .hdf5 conteniendo el dataset de angulos y coordenadas de GPS, elabora el posicionamiento 
    del drone con respecto al radar como origen en coordenadas cartesianas (X - Este, Y - Norte), y el de la
    esfera con respecto a la posicion del drone en base a los angulos roll y pitch leidos, retorna un 
    DataFrame de Pandas con las posiciones de drone y la esfera.

    * Variable PATH_HDF5_DRONE indica la ruta del archivo .hdf5 a leer
    * Se retorna un dataframe de Pandas

    '''

    L = self.cable_length - self.sphere_rad
    path_hdf5 = os.path.join(self.main_path,'results_exp/hdf5')

    def calc_dist(origin, end):

        R = 6372.8  # radio de la tierra en km

        lat_o = origin[0]; lon_o = origin[1]; alt_o = origin[2]
        lat_f = end[0]; lon_f = end[1]; alt_f = end[2]

        dLat = math.radians(lat_f - lat_o)
        dLon = math.radians(lon_f - lon_o)
        lat_o = math.radians(lat_o)
        lat_f = math.radians(lat_f)
    
        #Calculo del bearing
        diffLong = math.radians(lon_f - lon_o)
        x = math.sin(diffLong) * math.cos(lat_f)
        y = math.cos(lat_o) * math.sin(lat_f) - \
            (math.sin(lat_o) * math.cos(lat_f) * math.cos(diffLong))
        
        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        bearing = (initial_bearing + 360) % 360

        #Calculo de la distancia entre punto y punto
        a = math.sin(dLat / 2)**2 + math.cos(lat_o) * \
            math.cos(lat_f) * math.sin(dLon / 2)**2
        c = 2 * math.asin(math.sqrt(a))

        distance = R * c

        # Calculo de la distancia en cada eje dependiendo del bearing y la distancia p-p
        distX = distance * math.sin(math.radians(bearing)) * 1000  #metros
        distY = distance * math.cos(math.radians(bearing)) * 1000
        distZ = alt_f - alt_o
        
        return (distX, distY, distZ)

    rows = []

    for file in os.listdir(path_hdf5):
        h5log = h5py.File(path_hdf5+"/"+file, mode='r')

        for reg in h5log["dset"]:
            
            #datetm = datetime(int(reg[1]), int(reg[2]), int(reg[3]), int(reg[4]), int(reg[5]), int(reg[6]), int(reg[7]))
        

            coords_drone = (reg[8], reg[9], reg[10])   # Lat, Long, Alt del drone
            dist_drone = calc_dist(self.SOPHY_pos, coords_drone)
            
            roll = reg[11]
            pitch = reg[12]
            yaw = reg[13] # Roll, Pitch, Yaw        
            
            roll = math.radians(float(roll))
            pitch = math.radians(float(pitch))


            if((roll != 0.0) and (pitch != 0.0) and (reg[1] >= 2022)):

                #datetm = datetime(int(reg[1]), int(reg[2]), int(reg[3]), int(reg[4]), int(reg[5]), int(reg[6]), int(reg[7]))
                datetm = datetime(int(reg[1]), int(reg[2]), int(reg[3]), int(reg[4]), int(reg[5]), int(reg[6]))
                # Algoritmo de posicionamiento

                # Relacion entre a y b
                a = math.cos(pitch)/math.cos(roll)
                #print(a)

                #Se halla A en terminos de b
                A = math.sqrt( (a*math.sin(roll))**2 + (a*math.sin(pitch))**2)
                #print(A)

                #Se halla theta
                B = math.cos(pitch)
                theta = math.atan(A/B)
                #print(math.degrees(theta))

                #Se halla A y B
                A1 = L*math.sin(theta)
                B1 = L*math.cos(theta)
                #print(A1, B1)

                #Hallamos phi
                D = math.sin(pitch)
                C = math.sin(roll)
                phi = math.atan(D/C)
                #print(phi)

                #Correccion
                delta = phi

                #E y F
                E = A1*math.cos(delta)
                F = A1*math.sin(delta)

                x_esf = dist_drone[0] + F
                y_esf = dist_drone[1] + E
                z_esf = dist_drone[2] - B1

                range_dro = math.sqrt((dist_drone[0])**2 + (dist_drone[1])**2)
                range_esf = math.sqrt(x_esf**2 + y_esf**2)

                pos_esf = (x_esf, y_esf, z_esf)


                rows.append({"timestamp":datetm, 
                            "x_drone":dist_drone[0],
                            "y_drone":dist_drone[1],
                            "z_drone":dist_drone[2],
                            "r_drone":range_dro,
                            "x_esfera":x_esf,
                            "y_esfera":y_esf,
                            "z_esfera":z_esf,
                            "r_esfera":range_esf})
                
    return pd.DataFrame(rows)      
        
  def get_time_for_h5f(self,PATH):
      
      h5f = h5py.File(PATH,'r')
      utc_time = datetime.fromtimestamp(h5f['Data']['time'][0]).replace(microsecond=0)  #Timestamp
      return utc_time
  
  def get_powerH(self,PATH):
      
      h5f = h5py.File(PATH,'r')
      hPower = np.array(h5f['Data']['power']['H'])  #Canal H
      return hPower
  
  def get_powerV(self,PATH):
      
      h5f = h5py.File(PATH,'r')
      vPower = np.array(h5f['Data']['power']['V'])  #Canal V
      return vPower
  
  def get_elevation_h5f(self,PATH):
      
        h5f = h5py.File(PATH,'r')
        elevation = np.array(h5f['Metadata']['elevation'])  #Elevacion
        return elevation
  
  def get_range_h5f(self, PATH):
      
        h5f = h5py.File(PATH,'r')
        range = np.array(h5f['Metadata']['range'])  #Rango
        return range
  
  def get_azimuth_h5f(self, PATH):
        h5f = h5py.File(PATH,'r')
        azimuth = np.array(h5f['Metadata']['azimuth']).mean()
        return azimuth
  
  def is_sphere_drone_detected(self,dPower):
      
      """
      Retorna True si del dataset se reconocen ecos correspondientes al drone y la esfera separados por un espacio 
      (ecos de potencia menor). Retorna False si no se reconoce el patron ecos fuertes - espacio de ecos debiles - ecos fuertes
      dentro del dataset. 
      
      """
      
      n_gaps = 0
      end = False

      dPowerdB = 10*np.log10(dPower)           # dataset en dBm
      
      
      x,y = np.where(dPowerdB[self.idx_exp_ele:,self.idx_min_range_bin:self.idx_min_range_bin + 4] > self.threshold)  # filtro de valores de potencia
      
      x += self.idx_exp_ele 
      y += self.idx_min_range_bin

      x = list(sorted(set(x)))                 # se guardan los perfiles con valores de potencia sospechosos (drone y esfera)


      for i in range(len(x)):
              
          try:
            #si hay perfiles consecutivos con valores de potencia reconocibles 
            if ((x[i] + 1) == x[i+1]):
                #print("next")
                if (n_gaps >= 1):
                    end = True
            else:
                #Se detecta el espacio entre el drone y la esfera
                #print("gap")
                n_gaps += 1 
          
          except:
              pass
     
      return end
      

            

  def get_sphere_drone_echoes(self, power, ele, ran, azi ):
    
     
      powerdBm = 10*np.log10(power)           # dataset en dBm

      
      rows, cols = np.where(powerdBm[self.idx_exp_ele:,self.idx_min_range_bin:self.idx_min_range_bin + 4] > self.threshold)  # filtro de valores de potencia
      rows += self.idx_exp_ele
      cols += self.idx_min_range_bin

      rows_single = sorted(list(rows))
      #print(rows, cols)
    
      #print(rows_single)
      
      idx_separation = 0

      for i in range(len(rows_single)):
              
          try:

            if ( (rows_single[i] + 1) != rows_single[i+1] and (rows_single[i] != rows_single[i+1])):
                idx_separation = i
          
          except:
              pass


      profiles_drone = list(set(rows_single[:idx_separation+1]))
      profiles_sphere = list(set(rows_single[idx_separation+1:]))
    
      doper = np.array([rows, cols])


      drone_echos = []
      esfera_echos = []

      #Chequea los datos de cada eco asociado al drone y esfera
      
      for echo in doper.transpose():
 
          power_echo_db = powerdBm[echo[0]][echo[1]]
          power_echo_lnr = 10**(power_echo_db/10)
          range_echo = ran[echo[1]]
          ele_echo = ele[echo[0]]
          r_echo_limits = (round(range_echo,3)*1000 - self.range_resolution/2, round(range_echo,3)*1000 + self.range_resolution/2)
          

          if(echo[0] in profiles_drone):
              
              #print("Drone", round(power_echo,2), 10**(power_echo/10), ele_echo, r_echo_limits)
              drone_echos.append({"power_echo_db_H": power_echo_db,
                                  "power_echo_lnr_H": power_echo_lnr, 
                                  "ele_echo": ele_echo, 
                                  "range_limits":r_echo_limits, 
                                  "azimuth":azi,
                                  "coord":(echo[0], echo[1])})
              
          elif(echo[0] in profiles_sphere):
        
              #print("Esfera", round(power_echo,2), 10**(power_echo/10), ele_echo, r_echo_limits)
              esfera_echos.append({"power_echo_db_H": power_echo_db,
                                   "power_echo_lnr_H": power_echo_lnr, 
                                   "ele_echo": ele_echo, 
                                   "range_limits":r_echo_limits, 
                                   "azimuth":azi,
                                   "coord":(echo[0], echo[1])})
               
      
      df_drone = pd.DataFrame(drone_echos)
      df_esfera = pd.DataFrame(esfera_echos)

      return df_drone, df_esfera
      
     

  def get_compl_channel_echoes(self, idx_echoes, powerV):
      
      channel1 = []

      for echo in idx_echoes:
          channel1.append({"power_echo_db_V":10*np.log10(powerV[echo[0]][echo[1]]),
                           "power_echo_lnr_V":powerV[echo[0]][echo[1]]})
      

      df_channel01 = pd.DataFrame(channel1)

      return df_channel01
  
  def get_max_echo(self, echoes):
      
    idx_max = echoes["power_echo_lnr_H"].idxmax()
    return echoes.loc[idx_max]
  
  def get_weighting_functions(self, pos_esf, pos_dro, esf_echo):
      
      sigma_r = 0.35*self.pulse_width*speed_of_light/2
      sigma_xy = math.degrees(self.beam_width)/2.36
      OFFSET = 33.5                                       # Valor de offset angular de compensación de Yaw (Azimutal)
      
      r = pos_esf[3]                                      # Rango hasta el target (Directo)
      z = pos_esf[2]                                      # Altura del target (msnm) (Proyeccion y)  
      l = math.sqrt(r**2-z**2)                            # Rango horizontal (Proyeccion x)

      gamma = np.rad2deg(np.arctan((pos_dro[1])/(pos_dro[0])))   
      theta =  OFFSET - gamma
      #alfa =  np.rad2deg(np.arctan(f_esf/l))

      r_o = (esf_echo['range_limits'][0] + esf_echo['range_limits'][1]) / 2 - self.range_offset

      theta_X_bar = esf_echo["azimuth"]
      theta_Y_bar = esf_echo["ele_echo"] 

      theta_X  = theta 
      theta_Y = math.degrees(math.atan(z/l))

      Wr = math.exp(-((r-r_o)**2)/(2*sigma_r**2))
      Wb = math.exp(-((theta_X-theta_X_bar)**2)/(2*sigma_xy**2) -((theta_Y-theta_Y_bar)**2)/(2*sigma_xy**2))

      #print("theta_x: ", theta_X, "\ttheta_x_bar: ", theta_X_bar)
      #print("theta_y: ",theta_Y, "\ttheta_y_bar: ", theta_Y_bar)
      #print("r", r,"\tro", r_o)
      #print(theta_Y_bar)

      return Wr, Wb
  
  def get_hardt_constant(self, echo_sphere, pos_esf, rwf, bwf):
      
    if(isinstance(echo_sphere, pd.Series)):
        r_power = echo_sphere["power_echo_lnr_H"]/100

    else:
        r_power = echo_sphere/100

    range = pos_esf[3]
    gLNA_lnr = 10**(self.gLNA/10)

    return  ((r_power * range**4)/(self.pT*self.sigma*gLNA_lnr*rwf*bwf))
  
  def get_soft_constant(self, htc):

    lambda_sophy = speed_of_light/self.freq_oper

    return (16 * math.log(2) * lambda_sophy**4 * 10**18)/( htc * speed_of_light * math.pi * self.beam_width**2 * math.pi**5 * self.k_M)
  
  def remove_failed_profiles(self, dPower):
      
      '''
      Remueve los perfiles fallados (perfiles con un valor atipico de potencia en todos los range bins).
      Entrada --> Dataset original
      Salida --> Dataset filtrado

      '''

      first_from_row = dPower[:,0]
      idx_fail = np.where(np.isnan(first_from_row) == False)

      for idx in idx_fail[0]:
         dPower[idx,:] = 1e-7


      idx_wrong = np.where(np.isnan(first_from_row) == False)
      return dPower

  def get_difference_between_channels(self):
     
     '''
     Retorna la diferencia en los niveles de potencia [dBs] del pulso de transmision entre ambos canales.  
     '''
      
     path_h5 = os.path.join(self.hdf5_path,'S_RHI_AZ_28.0')

     first_file = os.listdir(path_h5)[self.idx_file]

     powerH = self.get_powerH(os.path.join(path_h5,first_file))
     powerV = self.get_powerV(os.path.join(path_h5,first_file))
     
     mean_H = 10*np.log10(np.mean(powerH[:,self.idx_t_conm]))
     mean_V = 10*np.log10(np.mean(powerV[:,self.idx_t_conm]))


     return mean_H - mean_V   

     #print(mean_H, mean_V, mean_H - mean_V)

  def plot_rhi(self, diff = 0):
     
     'Plotea el RHI leyendo archivos hdf5 de un directorio root,'
     
     a = 6374                 #Radio de la tierra
     ae = 4/3*a       
     
     #Crea el directorio de guardado de plots si no existe
     if os.path.isdir(os.path.join(self.main_path,'RHI_PLOTS')) == False:
           os.mkdir(os.path.join(self.main_path,'RHI_PLOTS'))

     for i in self.az_range:
         if os.path.isdir(os.path.join(self.main_path,'RHI_PLOTS',f'S_RHI_AZ_{i}')) == False:
             os.mkdir(os.path.join(self.main_path,'RHI_PLOTS',f'S_RHI_AZ_{i}'))

         if os.path.isdir(os.path.join(self.main_path,'RHI_PLOTS',f'S_RHI_AZ_POWER_CORR_{i}')) == False:
             os.mkdir(os.path.join(self.main_path,'RHI_PLOTS',f'S_RHI_AZ_POWER_CORR_{i}'))
    

     for fol in sorted(os.listdir(self.hdf5_path)):
        for file in sorted(os.listdir(self.hdf5_path+ "/" + fol)):
           
           path_h5 = os.path.join(self.hdf5_path, fol ,file)
           powerH = self.get_powerH(path_h5)
           powerV = self.get_powerV(path_h5) 

           utc_time = self.get_time_for_h5f(path_h5)
           az_file = file[-11:-7]

           n_ele = np.array(self.get_elevation_h5f(path_h5))
           n_ran = np.array(self.get_range_h5f(path_h5))
           n_azi = np.array(self.get_azimuth_h5f(path_h5))

           
           text_H = f"RHI Power at AZ: {az_file} CH 0 {utc_time} UTC-5"
           text_V = f"RHI Power at AZ: {az_file} CH 1 {utc_time} UTC-5"

           #Volumenes de resolucion en RHI

           r2, el_rad2 = np.meshgrid(n_ran, n_ele/180*math.pi)

           r21 = r2[:,:]
           el_rad21 = el_rad2[:,:]

           ads = np.multiply(r21, np.sin(el_rad21))
           ads2 = np.multiply(r21, np.cos(el_rad21))
          
           y = (r21**2 + ae**2 + 2*ads*ae)**0.5 - ae
           x = ae*np.arcsin(np.divide(ads2, ae +y))


           powerH = 10*np.log10(powerH)      # dataset en dBm
           powerV = 10*np.log10(powerV)      # dataset en dBm

           powerV_corr = powerV + diff     # dataset en dBm


           fig, (ax1, ax2) = plt.subplots(1,2,figsize=(13,6))

           plo1 = ax1.pcolormesh(x, y, powerH, shading='auto', vmin=-60, vmax=-20, edgecolors='k', linewidths=self.linewidth_plot, cmap="jet")

           ax1.set(ylim=(0,0.15))
           ax1.set(xlim=(0,0.5))
           ax1.set_title(text_H)
           ax1.set_xlabel("Distance from radar [km]")
           ax1.set_ylabel("Height [km]")

           plo2 = ax2.pcolormesh(x, y, powerV, shading='auto', vmin=-60, vmax=-20, edgecolors='k', linewidths=self.linewidth_plot, cmap="jet")

           ax2.set(ylim=(0,0.15))
           ax2.set(xlim=(0,0.5))
           ax2.set_title(text_V)
           ax2.set_xlabel("Distance from radar [km]")
           ax2.set_ylabel("Height [km]")

           fig.colorbar(plo1, ax = ax1, orientation='vertical', label='dBm' )
           fig.colorbar(plo2, ax = ax2, orientation='vertical', label='dBm' )

           date, time = str(utc_time).split(' ')

           date = date.replace("-", "")
           time = time.replace(":", "")
           
          
           plt.savefig(os.path.join(self.main_path, 'RHI_PLOTS',f'S_RHI_AZ_{az_file}', f'SOPHY_{date}_{time}_A{az_file}_S.png'))

           fig, (ax1, ax2) = plt.subplots(1,2,figsize=(13,6))

           plo1 = ax1.pcolormesh(x, y, powerH, shading='auto', vmin=-20, vmax=80, edgecolors='k', linewidths=self.linewidth_plot, cmap="jet")
 

           ax1.set(ylim=(0,0.15))
           ax1.set(xlim=(0,0.5))
           ax1.set_title(text_H)
           ax1.set_xlabel("Distance from radar [km]")
           ax1.set_ylabel("Height [km]")

           plo2 = ax2.pcolormesh(x, y, powerV_corr, shading='auto', vmin=-60, vmax=-20, edgecolors='k', linewidths=self.linewidth_plot, cmap="jet")

           ax2.set(ylim=(0,0.15))
           ax2.set(xlim=(0,0.5))
           ax2.set_title(text_V)
           ax2.set_xlabel("Distance from radar [km]")
           ax2.set_ylabel("Height [km]")

           fig.colorbar(plo1, ax = ax1, orientation='vertical', label='dBm' )
           fig.colorbar(plo2, ax = ax2, orientation='vertical', label='dBm' )

           date, time = str(utc_time).split(' ')

           date = date.replace("-", "")
           time = time.replace(":", "")
           

           plt.savefig(os.path.join(self.main_path, 'RHI_PLOTS',f'S_RHI_AZ_POWER_CORR_{az_file}', f'SOPHY_{date}_{time}_A{az_file}_S.png'))

  def proc_experiment(self, diff, df):
    
      """
      Procesa cada archivo hdf5 juntando el dataset de posicion de la esfera de acuerdo a la marca de tiempo, reconoce y separa 
      los ecos correspondientes al drone y la esfera, finalmente
      """

      tmp_cont = []
      htc_H_cont = []
      htc_V_wo_corr_cont = []
      htc_V_corr_cont = []
      stc_H_cont = []
      stc_V_wo_corr_cont = []
      stc_V_corr_cont = []

      #Por cada archivo de cada directorio (1 por cada hora)
      for fol in sorted(os.listdir(self.hdf5_path)):
          for file in sorted(os.listdir(self.hdf5_path+"/"+fol)):             
            
            path_h5 = os.path.join(self.hdf5_path,fol,file) 
            powerH = self.get_powerH(path_h5)
            powerV = self.get_powerV(path_h5)

            elevation = self.get_elevation_h5f(path_h5)
            range_arr = self.get_range_h5f(path_h5)
            azimuth = self.get_azimuth_h5f(path_h5)
            tmstamp = self.get_time_for_h5f(path_h5)

            #Eliminando los perfiles fallidos
            powerH_corr = self.remove_failed_profiles(powerH)
            powerV_corr = self.remove_failed_profiles(powerV)
            
            #powerV_corr_power = powerV_corr + (10**(diff/10))
            powerV_corr_power = 10*np.log10(powerV_corr) + diff
            powerV_corr_power = 10**(powerV_corr_power/10)

            #Si se detectan ecos correspondientes al drone y a la esfera:
            if(self.is_sphere_drone_detected(powerH_corr)):
                
                #self.get_sphere_drone_echoes(powerH_corr, elevation, range_arr, azimuth)
                droneH, sphereH = self.get_sphere_drone_echoes(powerH_corr, elevation, range_arr, azimuth)
                #print(elevation.shape)

              
                droneV_wo_corr = self.get_compl_channel_echoes(droneH["coord"], powerV_corr)
                sphereV_wo_corr = self.get_compl_channel_echoes(sphereH["coord"], powerV_corr)

                droneV_corr = self.get_compl_channel_echoes(droneH["coord"], powerV_corr_power)
                sphereV_corr = self.get_compl_channel_echoes(sphereH["coord"], powerV_corr_power)

                df_wo_corr = pd.concat([sphereH, sphereV_wo_corr], axis = 1)
                df_corr = pd.concat([sphereH, sphereV_corr], axis = 1)

                #print(df_wo_corr)

                #Se obtiene el eco de maxima potencia
                max_echo_drone_H = self.get_max_echo(droneH)
                max_echo_sphere_H = self.get_max_echo(sphereH)

                #Se obtiene el eco correspondiente al canal complementario (CH1)

                max_echo_sphere_V_lnr_wo_corr = powerV_corr[max_echo_sphere_H['coord'][0], max_echo_sphere_H['coord'][1]]
                max_echo_sphere_V_db_wo_corr = 10*np.log10(max_echo_sphere_V_lnr_wo_corr)

                max_echo_sphere_V_lnr_corr = powerV_corr_power[max_echo_sphere_H['coord'][0], max_echo_sphere_H['coord'][1]]
                max_echo_sphere_V_db_corr = 10*np.log10(max_echo_sphere_V_lnr_corr)

                max_echo_drone_V_lnr_wo_corr = powerV_corr[max_echo_drone_H['coord'][0], max_echo_drone_H['coord'][1]]
                max_echo_drone_V_db_wo_corr = 10*np.log10(max_echo_drone_V_lnr_wo_corr)

                max_echo_drone_V_lnr_corr = powerV_corr_power[max_echo_drone_H['coord'][0], max_echo_drone_H['coord'][1]]
                max_echo_drone_V_db_corr = 10*np.log10(max_echo_drone_V_lnr_corr)

                #print(max_echo_sphere_H, file )

                azi_pos = max_echo_drone_H["azimuth"]
                
                #Busca una entrada en el dataframe del drone/esfera 
                df_second = df.loc[ df["timestamp"] == tmstamp,:]
                #print(df["timestamp"], tmstamp)

                #Se promedian todas las coincidencias en dicho segundo
                x_drone_mean = df_second['x_drone'].mean()
                y_drone_mean = df_second['y_drone'].mean()
                z_drone_mean = df_second['x_drone'].mean()
                r_drone_mean = df_second['r_drone'].mean()
                x_esf_mean = df_second['x_esfera'].mean()
                y_esf_mean = df_second['y_esfera'].mean()
                z_esf_mean = df_second['z_esfera'].mean()
                r_esf_mean = df_second['r_esfera'].mean()

                pos_esf = (x_esf_mean, y_esf_mean, z_esf_mean, r_esf_mean)
                pos_dro = (x_drone_mean, y_drone_mean, z_drone_mean, r_drone_mean)

                rwf, bwf = self.get_weighting_functions(pos_esf, pos_dro, max_echo_sphere_H)

                #print(rwf,bwf)
                #print(df_second)

                htc_H = self.get_hardt_constant(max_echo_sphere_H, pos_esf, rwf, bwf)
                stc_H = self.get_soft_constant(htc_H)

                htc_V_wo_corr = self.get_hardt_constant(max_echo_sphere_V_lnr_wo_corr, pos_esf, rwf, bwf)
                stc_V_wo_corr = self.get_soft_constant(htc_V_wo_corr)

                htc_V_corr = self.get_hardt_constant(max_echo_sphere_V_lnr_corr, pos_esf, rwf, bwf)
                stc_V_corr = self.get_soft_constant(htc_V_corr)


                if rwf < 0.9:
                   
                    # H
                    htc_H_cont.append(htc_H)
                    stc_H_cont.append(10*np.log10(stc_H))
                    
                    # V sin correccion
                    htc_V_wo_corr_cont.append(htc_V_wo_corr)
                    stc_V_wo_corr_cont.append(10*np.log10(stc_V_wo_corr))

                    # V con correccio
                    htc_V_corr_cont.append(htc_V_corr)
                    stc_V_corr_cont.append(10*np.log10(stc_V_corr))
                    
                    # Tiempo
                    tmp_cont.append(tmstamp)
                    


      return({"H":{"HTC": htc_H_cont, "STC": stc_H_cont}, "V":{"sin_corr":{"HTC":htc_V_wo_corr_cont, "STC":stc_V_wo_corr_cont}, 
                                                               "con_corr":{"HTC":htc_V_corr_cont, "STC":stc_V_corr_cont}},
                                                          "timestamp": tmp_cont})
  
                
      

      