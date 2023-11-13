#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pywencai
import qstock as qs


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import coint_johansen,VECM
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')


# In[27]:


codes = ['沪深300ETF','国债ETF','大宗商品ETF']
df = qs.get_price(codes,end='20230619').dropna()
def test_stationarity(timeseries):
    dftest = adfuller(timeseries,autolag='AIC')
    return dftest[1] < 0.05
def difference_until_stationary(df):
    while not df.apply(test_stationarity):
        df = df.diff().dropna()
    return df

returns = np.log(df/df.shift(1)).dropna()


# In[42]:


model = VAR(endog = returns)
lags = range(1,10)
criterion = 'aic'
criteria = []
for lag in lags:
    result = model.fit(lag)
    criteria.append(result.info_criteria[criterion])

best_lag = lags[criteria.index(min(criteria))]

results = model.fit(best_lag)
print(results.pvalues)


# In[44]:


def show_result(df): 
    results_df = pd.DataFrame()
    
    for var in results.names:
        coeffs = results.params[var].round(3)
        pvalues = results.pvalues[var].round(3)
        for i in range(len(pvalues)):
            if pvalues[i] < 0.01:
                coeffs[i] = str(coeffs[i]) + '***'
            elif pvalues[i] < 0.05:
                coeffs[i] = str(coeffs[i]) + '**'
            elif pvalues[i] < 0.1:
                coeffs[i] = str(coeffs[i]) + '*'
        results_df[var + '_coeff'] = coeffs
        results_df[var + '_pvalue'] = pvalues
    return results_df
show_result(results)


# In[17]:


forecast = results.forecast(returns.values[-best_lag:],steps = 10)

irf = results.irf(10)
irf.plot(orth = True)


# In[21]:


def check_coint(df):
    s1 = df.iloc[:,0]
    s2 = df.iloc[:,1]
    coint_t,p_value,crit_value = coint(s1,s2)
    if p_value < 0.05:
        return True
    else:
        return False
    
def check_johansen(df):
    johansen_test = coint_johansen(df.values,det_order=0,k_ar_diff=1)
    if johansen_test.lr1[0] > johansen_test.cvt[0,1]:
        return True
    else:
        return False


# In[23]:


codes = ['sh','sz','sz50','cyb','hs300','zxb']
data = qs.get_price(codes,end='20230620',freq='w').dropna()[-200:]

fig,ax = plt.subplots(3,2,figsize=(12,10))
k=0
for i in range(3):
    for j in range(2):
        data.iloc[:,k].plot(ax=ax[i,j]);
        ax[i,j].set_title(data.columns[k]);
        k+=1
plt.tight_layout()

