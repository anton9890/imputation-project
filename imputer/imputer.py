import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
# from model.train import train
# from model.train import train_gan
# from model.train import train_dcgan
import seaborn as sns
import matplotlib.pyplot as plt
# from utils.util import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def fill_foward(data):
   df = data.fillna(method='ffill')
   return df

def moving_average(data, window=50, min_periods=1, inplace=False):
   temp = data.ffill()
   data = temp.rolling(window=window, min_periods=min_periods).mean()
   return data

def moving_min(data, window=50, min_periods=1, inplace=False):
   # temp = data.ffill()
   data = data.rolling(window=window, min_periods=min_periods).min()
   return data

def moving_max(data, window=50, min_periods=1, inplace=False):
   # temp = data.ffill()
   data = data.rolling(window=window, min_periods=min_periods).max()
   return data

def interpolated(data ,method='linear', limit_direction='forward'):
   data = pd.DataFrame(data)
   return data.interpolate(method=method, limit_direction=limit_direction)

def mice(data, estimator, max_iter):
    estimator_dict = {
        'linear': LinearRegression,
        'randomforest': RandomForestRegressor,
        'xgb': XGBRegressor,
        'lgbm': LGBMRegressor,
    }

    if estimator not in estimator_dict:
        raise ValueError("Invalid estimator. Please choose from 'linear', 'randomforest', 'xgb', 'lgbm'.")

    estimator = estimator_dict[estimator]()

    data = IterativeImputer(estimator=estimator,
                            missing_values=np.nan, 
                            max_iter=max_iter, 
                            verbose=2,
                            imputation_order='roman').fit_transform(data)
    return data

def moving_average_fix(data, window=85000, min_periods=1):
   # temp = data.ffill()
   temp = data.rolling(window=window, min_periods=min_periods).mean()
   filled_data = data.fillna(temp)
   return filled_data

def moving_min_fix(data, window=85000, min_periods=1):
   # temp = data.ffill()
   temp = data.rolling(window=window, min_periods=min_periods).min()
   filled_data = data.fillna(temp)
   return filled_data

def moving_max_fix(data, window=85000, min_periods=1):
   # temp = data.ffill()
   temp = data.rolling(window=window, min_periods=min_periods).max()  
   filled_data = data.fillna(temp)
   return filled_data

def moving_average_fix1(data, window=600, min_periods=1):
   # temp = data.ffill()
   for i in range(1000):
      temp = data.rolling(window=window, min_periods=min_periods).mean()
      data = data.fillna(temp)
   return data

def moving_min_fix1(data, window=600, min_periods=1):
   # temp = data.ffill()
   for i in range(1000):
      temp = data.rolling(window=window, min_periods=min_periods).min()
      data = data.fillna(temp)
   return data

def moving_max_fix1(data, window=600, min_periods=1):
   # temp = data.ffill()
   for i in range(1000):
      temp = data.rolling(window=window, min_periods=min_periods).max()
      data = data.fillna(temp)
   return data

def fill_foward_fix(data):
   data = data.copy()
   if data.iloc[0].isna().any():
      data.iloc[0] = data.mean(skipna=True)

   missing = data.isnull()

   # 각 연속된 결측치 그룹의 시작점과 길이 찾기
   starts = np.where(missing.diff() == 1)[0]
   lengths = np.diff(np.append(starts, missing.sum()))
   starts = starts[::2]
   lengths = lengths[::2]
   
   for start, length in zip(starts, lengths):
      try: 
         data.iloc[start:start+length] = data.iloc[start-length:start].values
      except:
         continue
   result = data.fillna(method='ffill')
   return result

# def fill_ae(df, window=10300, batch_size=64, lr=0.0002, b1=0.5, b2=0.999, n_epochs=1):   
#    temp = df.copy()
#    org_df = temp.iloc[:-window]
#    org_arr = org_df.values

#    # temp = outlier(df)
#    temp = fill_foward_fix(temp)
#    gen = train(temp, window, batch_size, lr, b1, b2, n_epochs)
   
#    result = np.where(np.isnan(org_arr), gen, org_arr)
#    # result = org_df.fillna(gen)
   
#    org_df.iloc[:, :] = result

#    mse, rmse, mae, r2 = metrics(temp[:-window].values ,gen)
#    print("실제데이터:생성데이터 평가")
#    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

#    mse, rmse, mae, r2 = metrics(temp[:-window].values ,result)
#    print("실제데이터:보간데이터 평가")
#    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

#    # plt.plot(temp.values)
#    # plt.plot(gen)
#    # plt.show()
#    return result

# def fill_gan(df, window=10300, batch_size=64, lr=0.0002, b1=0.5, b2=0.999, n_epochs=1, latent_dim=500):   
#    temp = df.copy()
#    org_df = temp.iloc[:-window]
#    org_arr = org_df.values

#    # temp = outlier(df)
#    #temp = fill_foward_fix(temp)
#    temp = temp.fillna(0)
#    gen = train_gan(temp, window, batch_size, lr, b1, b2, n_epochs, latent_dim)
   
#    result = np.where(np.isnan(org_arr), gen, org_arr)
#    # result = org_df.fillna(gen)
   
#    org_df.iloc[:, :] = result

#    mse, rmse, mae, r2 = metrics(temp[:-window].values ,gen)
#    print("실제데이터:생성데이터 평가")
#    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

#    mse, rmse, mae, r2 = metrics(temp[:-window].values ,result)
#    print("실제데이터:보간데이터 평가")
#    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

#    # plt.plot(temp.values)
#    # plt.plot(gen)
#    # plt.show()
#    return result

# def fill_dcgan(df, window=10300, batch_size=64, lr=0.0002, b1=0.5, b2=0.999, n_epochs=1, latent_dim=500):   
#    temp = df.copy()
#    org_df = temp.iloc[:-window]
#    org_arr = org_df.values

#    # temp = outlier(df)
#    #temp = fill_foward_fix(temp)
#    temp = temp.fillna(0)
#    gen = train_dcgan(temp, window, batch_size, lr, b1, b2, n_epochs, latent_dim)
   
#    result = np.where(np.isnan(org_arr), gen, org_arr)
#    # result = org_df.fillna(gen)
   
#    org_df.iloc[:, :] = result

#    mse, rmse, mae, r2 = metrics(temp[:-window].values ,gen)
#    print("실제데이터:생성데이터 평가")
#    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

#    mse, rmse, mae, r2 = metrics(temp[:-window].values ,result)
#    print("실제데이터:보간데이터 평가")
#    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

#    # plt.plot(temp.values)
#    # plt.plot(gen)
#    # plt.show()
#    return result

def mean(data):
   data = pd.DataFrame(data)
   data = data.fillna(data.mean())
   return data.values
