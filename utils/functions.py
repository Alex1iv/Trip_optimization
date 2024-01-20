# Custom functions
import pandas as pd
import numpy as np
import os

from utils.config_reader import config_reader 
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, roc_auc_score
from zipfile import ZipFile

import matplotlib.pyplot as plt



# # Импортируем константы из файла config
config = config_reader('../config/config.json')

def read_csv_from_zip(path:str=config.data_dir, exclusion:dict=None, index_col=None, ): #sep=None, filename=None, 
    
    mounts = dict()
    
    #lambda sep: ';' if sep is None else sep
    
    for archive in os.listdir(path):
        if archive in exclusion.keys():
            for filename, sep in exclusion.items():
                with ZipFile(os.path.join(config.data_dir, archive)) as myzip:
                    for file in myzip.namelist():
                        mounts[f"{filename[:-4]}".format(filename)] =  pd.read_csv(myzip.open(filename[:-4]+'.csv'), sep=sep)
        else:
            with ZipFile(os.path.join(config.data_dir, archive)) as myzip: 
                for file in myzip.namelist():
                    mounts[f"{file[:-4]}".format(file)] = pd.read_csv(myzip.open(file))             
        
                    
    return mounts

# def read_csv_from_zip1(path:str=config.data_dir, exclusion:list=None, index_col=None, ): #sep=None, filename=None, 
    
#     mounts = dict()
    
#     #lambda sep: ';' if sep is None else sep
    
#     for archive in os.listdir(path):
#         if archive not in exclusion:
#             with ZipFile(os.path.join(config.data_dir, archive)) as myzip: #as myzip
#                 for file in myzip.namelist():
#                     mounts[f"{file[:-4]}".format(file)] = pd.read_csv(myzip.open(file)) #, index_col=0 
#         # для файлов с сепаратором ';'            
#         else:
#             with ZipFile(os.path.join(config.data_dir, archive)) as myzip:
#                 for file in myzip.namelist():
#                     mounts[f"{file[:-4]}".format(file)] =  pd.read_csv(myzip.open(file), sep=';')
                    
#     return mounts


def add_datetime_features(data:pd.DataFrame)->pd.DataFrame:
    """Добавляет 3 признака: дата включения счетчика - начала поездки (без времени),\
        час дня включения счетчика,  наименование дня недели, в который был включен счетчик

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    data['pickup_date'] = data['pickup_datetime'].dt.date
    data['pickup_hour'] = data['pickup_datetime'].dt.hour
    data['pickup_day_of_week'] = data['pickup_datetime'].dt.dayofweek 
    data['pickup_day_of_week'] = data['pickup_day_of_week'].astype('category').cat\
        .rename_categories({0:'Mon',1:'Tue',2:'wed', 3:'Thu',4:'Fri',5:'Sat',6:'Sun'}).astype('category')
        
    return data


def add_holiday_features(data:pd.DataFrame, holiday_data:pd.DataFrame)->pd.DataFrame:
    """ Добавляем признак совершения поездки в праздничный день (1 - да, 0 - нет). 

    Args:
        data (pd.DataFrame): датафрейм с даннными о поездках
        holidays (pd.DataFrame): массив с датами праздников США

    Returns:
        pd.DataFrame: массив даннных о поездках с признаком поездки в праздник
    """
    # даты праздников
    days_off = holiday_data['date'].values 
    
    # даты в строковый формат
    data['pickup_date_str'] = data['pickup_date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    data['pickup_holiday'] = data['pickup_date_str'].apply(lambda x: 1 if x in days_off else 0)
    data.drop(['pickup_date_str'], axis=1, inplace=True)
             
    return data


def add_osrm_features(data:pd.DataFrame, osrm_data:pd.DataFrame)->pd.DataFrame:
    """ Объединяет 2 датафрейма. В результате добавляет к data признаки: 
            total_distance,  total_travel_time, number_of_steps.

    Args:
        data (pd.DataFrame): датафрейм с даннными о поездках
        osrm_data (pd.DataFrame): датафрейм с даннными о кратчайшем пути

    Returns:
        pd.DataFrame: объединённый датафрейм, в который добавлены признаки : \
            total_distance,  total_travel_time, number_of_steps.
    """    
    data = data.merge(osrm_data, how='left', on='id')
    return data


def get_haversine_distance(lat1, lng1, lat2, lng2):
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # радиус земли в километрах
    EARTH_RADIUS = 6371 
    # считаем кратчайшее расстояние h по формуле Хаверсина
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def get_angle_direction(lat1, lng1, lat2, lng2):
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # считаем угол направления движения alpha по формуле угла пеленга
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha

def add_geographical_features(data:pd.DataFrame)->pd.DataFrame:
    """Ддобавляет 2 признака в неё двумя столбцами:

    haversine_distance — расстояние Хаверсина между точками влючёния и выключения счетчика;
    direction — направление движения между точками влючёния и выключения счетчика;
    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    data['haversine_distance'] = data.apply(lambda x: get_haversine_distance(\
        x['pickup_latitude'],  x['pickup_longitude'], \
        x['dropoff_latitude'], x['dropoff_longitude']), axis=1)
    
    data['direction'] = data.apply(lambda x: get_angle_direction(\
        x['pickup_latitude'],  x['pickup_longitude'], \
        x['dropoff_latitude'], x['dropoff_longitude']), axis=1)
                         
    return data

def add_cluster_features(data:pd.DataFrame, cluster)->pd.DataFrame:
    """Add cluster feature to dataframe

    Args:
        data (pd.DataFrame): _description_
        cluster (_type_):cluster feature

    Returns:
        (pd.DataFrame): joined dataframe
    """    
    coordinats = np.hstack((data[['pickup_latitude', 'pickup_longitude']], data[['dropoff_latitude', 'dropoff_longitude']]))

    data['geo_cluster'] = cluster.predict(coordinats)
    return data

def add_weather_features(data:pd.DataFrame, weather:pd.DataFrame)->pd.DataFrame:
    """add weather feature

    Args:
        data (pd.DataFrame): original dataframe
        weather (_type_): frature from the weather dataframe

    Returns:
        pd.DataFrame: joined dataframe
    """    
    weather['time'] = pd.to_datetime(weather['time'])
    weather['date'] = weather['time'].dt.date
    weather['hour'] = weather['time'].dt.hour

    data = pd.merge(data, weather[[\
        'date', 'hour', 'temperature', 'visibility', 'wind speed', 'precip', 'events']],
        left_on=['pickup_date','pickup_hour'], right_on=['date', 'hour'], how='left')

    data = data.drop(columns=['date', 'hour'], axis=1)
 
    return data


def fill_null_weather_data(data:pd.DataFrame)->pd.DataFrame:
    """Fill missed weather data

    Returns:
        (pd.DataFrame): fully filled dataframe without missing data entries
    """    
    for col in ['temperature', 'visibility', 'wind speed', 'precip']:
        data[col].fillna(data.groupby('pickup_date')[col].transform('median'), inplace=True)
    
    for col in ['total_distance', 'total_travel_time', 'number_of_steps']:
        data[col].fillna(data[col].median(), inplace=True)
        
    data['events'].fillna('None', inplace=True)
     
    return data

history = dict()

def plot_history(history:dict=history, model_name:str=None, plot_counter:int=None):
    """Training history visualization
    
    Аргументы:
    history (keras.callbacks.History) - Training history data,
    model_name (str) - figure title. Use: model.name
    plot_counter (int) - figure id.      
    """
    mse_metric = history.history['mse'] 
    mse_val = history.history['val_mse']  # validation sample
        
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(mse_metric))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

    ax[0].plot(epochs, loss, 'b', label='Training')
    ax[0].plot(epochs, val_loss, 'r', label='Validation')
    ax[0].set_xlabel('Epoch', size=11)
    ax[0].set_ylabel('Loss', size=11)
    ax[0].set_title('Loss')
    ax[0].legend(['train', 'val'])

    ax[1].plot(epochs, mse_metric, 'b', label='Training')
    ax[1].plot(epochs, mse_val, 'r', label='Validation')
    ax[1].set_xlabel('Epoch', size=11)
    ax[1].set_ylabel('MSE value', size=11)
    ax[1].set_title(f"MSE")
    ax[1].legend(['train', 'val'])

    if plot_counter is not None:
        plt.suptitle(f"Fig.{plot_counter} - {model_name} model", y=0.05, fontsize=14)
        #plt.savefig(config.PATH_FIGURES + f'fig_{plot_counter}.png')
    
    else: 
        plot_counter = 1
        plt.suptitle(f"Fig.{plot_counter} - {model_name} model", y=-0.1, fontsize=14)  
    plt.tight_layout();
