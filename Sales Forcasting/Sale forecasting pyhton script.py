#importing essenstial libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
data = pd.read_excel('Data.xlsx')
data.head()

#making a column Date_split which contain a list of character 'M' and date
data['Date_split'] = data.Date.str.split(' ')
#making a column Date2 which contain only date from date column
data['Date2'] = data.Date_split.str.get(1)
# droping date and Date_split column
data.drop(['Date','Date_split'],axis=1,inplace=True)
data.head()

#changing date2 column into datetime from string
data['Date2'] = pd.to_datetime(data['Date2'],format='%m.%Y')

#checking for null values
data.apply(lambda x : sum(x.isnull()))
df_value = missingValue(data,'Product Code','Date2')
df_value.to_json('Missing Value Count.json')

#filling missing value by mean in Actual column
data['Actual'].fillna(data['Actual'].mean(),inplace=True)
data['Actual'].isnull().sum()

#checking for last and first date
print(data['Date2'].min())
print(data['Date2'].max())

# Setting index to date column
data = data.set_index('Date2')
data.drop('Product Code',axis=1,inplace=True)
data.head()

# Ploting actual data with time 
plt.figure(figsize=(15,8))
plt.plot(data['Actual'])
plt.title('Sales over Time')
plt.xlabel('Time')
plt.ylabel('Sales ($)')
plt.show()

# Grouping the average sale by year and seeing the trend
year_mean = data['Actual'].resample('Y').mean()
print(year_mean)
year_mean.plot()
plt.show()


#  calcualting average month sales
month_mean = data.resample('MS').mean()
month_mean['2017']

#plotting averge sale by month
plt.figure(figsize=(15,5))
plt.plot(month_mean.Actual)
plt.title('Average sale of company over Month')
plt.ylabel('Average Sale ($)')
plt.xlabel('month')
plt.show()

# Ploting Seasonality,trend and residual of average sale of month
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(month_mean.Actual, model='additive')
fig = decomposition.plot()
plt.show()

checking_Stationary(data.Actual,lags=30)


import statsmodels.api as sm
import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


#Parameter selection
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(month_mean, order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
mod1 = sm.tsa.statespace.SARIMAX(month_mean,order=(1, 1, 1),seasonal_order=(1, 1, 1, 12),enforce_stationarity=False, enforce_invertibility=False)
results1 = mod1.fit()
print(results1.summary())        


pred = results1.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = month_mean['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()


pred_uc = results.get_forecast(steps=10)
pred_ci = pred_uc.conf_int()
ax = month_mean.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel(' Sales')
plt.legend()
plt.show()


print('Predacted value of Average sales for Oct,2019 is {}'.format(pred_uc.predicted_mean['2019-10-01']))

pred_uc.predicted_mean.to_csv('Prediction.csv')
class Sales:
    def MAPE(y_true,y_pred):
        ''' Give the mean absloute precentage error'''
        mape = np.mean(np.abs(y_true - y_true)/y_true)
        if mape> 1:
        mape=1
        return mape 

 
    def missingValue(df,col1,col2):
        ''' Give the count of col2 for missing value group by col1'''
        value = df[df.isnull().any(axis=1)].reset_index()
        df_group = value.groupby(col1)[col2].value_counts()
        return df_group

       #Checking stationarity of the data
   import statsmodels.tsa.api as smt
   def checking_Stationary(y, lags=None, figsize= (12,7),syle='bmh'):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        with plt.style.context(style='bmh'):
           fig = plt.figure(figsize=figsize)
           layout = (2,2)
           ts_ax = plt.subplot2grid(layout, (0,0),colspan=2)
           acf_ax = plt.subplot2grid(layout, (1,0))
           pacf_ax = plt.subplot2grid(layout, (1,1),colspan=2)
        
         y.plot(ax=ts_ax)
         p_value = sm.tsa.stattools.adfuller(y)[1]
         ts_ax.set_title('Time Series Analysis Plot\n Dickey-Fuller: p={0:.5f}'.format(p_value))
         smt.graphics.plot_acf(y,lags=lags,ax=acf_ax)
         smt.graphics.plot_pacf(y,lags=lags,ax=pacf_ax)
         plt.tight_layout()
