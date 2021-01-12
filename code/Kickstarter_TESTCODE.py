#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:12:35 2020

@author: duncanwang
"""

#INSY 662 -- Individual Project - For Grading
#Duncan Wang -- 260710229
#11/29/2020

########################## PART 1: DATA PREPROCESSING ######################

#import packages 
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display

#read grading file here!!!
df = pd.read_excel('Kickstarter-Test.xlsx')

#1.1 : HANDLE NULL VALUES_____________________________________________

#category column has many null values -- will impute since this may be a valuable category, fill null category values with new 'Other' values 
df['category'].fillna('Other',inplace = True)
#average imputation for name and blurb lengths, since there are only 5 values 
df['name_len'].fillna(df['name_len'].mean(),inplace = True)
df['name_len_clean'].fillna(df['name_len_clean'].mean(),inplace = True)
df['blurb_len'].fillna(df['blurb_len'].mean(),inplace = True)
df['blurb_len_clean'].fillna(df['blurb_len_clean'].mean(),inplace = True)
#this column has 66% null values, so will drop
df.drop(['launch_to_state_change_days'], axis = 1, inplace = True)

#1.2: DROP PREDICTORS FROM ANALYSIS____________________________________

#drop all observations other than success/failed in project state 
df.drop(df[df['state']== 'canceled'].index, inplace = True)
df.drop(df[df['state']== 'live'].index, inplace = True)
df.drop(df[df['state']== 'suspended'].index, inplace = True)

#drop project id and name, as they have no predictive value and length of name is already a predictor
#drop disable communication as there is only one instance of "false" (categories are imbalanced)
df.drop(['project_id','name','disable_communication'], axis = 1, inplace = True)
#drop date/time variables which have already been broken down into their subparts 
df.drop(['deadline','state_changed_at','created_at','launched_at'], axis = 1, inplace = True)
#drop day as time can be better represented by week/month/hour, and this is a high cardinality categorical variable that is hard to encode
df.drop(['deadline_day','created_at_day','launched_at_day'], axis = 1, inplace = True)

#drop variables that can only be observed after launch
df.drop(['pledged','backers_count', 'staff_pick','spotlight','state_changed_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr'], axis = 1, inplace = True)

#1.3: HANDLE MULTICOLLINEARITY________________________________________

#most projects are created, launched, and have their deadline at the same year. Since we have the difference between creation and launch and launch to deadline, we can get rid of these 
df.drop(['created_at_yr', 'deadline_yr'], axis = 1, inplace = True)
#name len and blurb len are highly correlated with their clean versions. We want to keep the clean version as these are more likely to contain key words, so drop the rest
df.drop(['name_len', 'blurb_len'], axis = 1, inplace = True)
#launched at hr and deadline hr are highly correlated -- possible that this is set together as num of days, therefore, drop deadline hr as it is not going to influence money raised much
#the hour the project is created at is also unlikely to influence the money raised 
df.drop(['deadline_hr', 'created_at_hr'], axis = 1, inplace = True)

#1.4: ADD NEW DERIVED PREDICTORS _______________________________________

#convert the goal variable to goal in USD by multiplying by static usd rate -- this allows values to be more comparable/on the same scale
df['goal_usd'] = round(df['goal']*df['static_usd_rate'],0)

#create a new variable to represent the ratio of goal in USD/how long the project launch period is
df['goal_to_launch_time_ratio'] = round(df['goal_usd']/df['launch_to_deadline_days'],3)
#create a new variable to represent the ratio of goal in USD/how long the project creation to launch period is, add 1 to handle division by 0 issues 
df['goal_to_create_time_ratio'] = round(df['goal_usd']/(df['create_to_launch_days']+1),3)

#drop the original goal variable 
df.drop(['goal'], axis = 1, inplace = True)

#1.5 CATEGORICAL ENCODING OF VARIABLES________________________________________

#reduce dimensionality of categorical predictors to prevent overfitting and cardinality issues 

#condense category into fewer sections 
df['category'].replace(['Comedy','Academic','Thrillers','Shorts','Places', 'Spaces','Makerspaces','Immersive','Experimental','Flight', 'Wearables'], 'Other',inplace = True)
df['category'].replace(['Webseries'],'Web', inplace = True) 
df['category'].replace(['Apps'],'Software', inplace = True) 
df['category'].replace(['Robots'],'Gadgets', inplace = True) 
df['category'].replace(['Plays','Musical','Sound','Festivals','Blues'],'Arts', inplace = True) 

#condense hr into periods of the day
df['launched_at_hr'].replace([1,2,3,4,5,6], 1, inplace = True)
df['launched_at_hr'].replace([7,8,10,11,12], 2, inplace = True)
df['launched_at_hr'].replace([13,14,15,16,17,18], 3, inplace = True)
df['launched_at_hr'].replace([19,20,21,22,23,0], 4, inplace = True)

#condense month into seasons
months = [df['deadline_month'],df['created_at_month'],df['launched_at_month']]
for i in months:
    i.replace([1,2,3,4], 1, inplace = True)
    i.replace([5,6,7,8], 2, inplace = True)
    i.replace([9,10,11,12], 3, inplace = True)

#condense days of the week into weekday/weekend
weeks = [df['deadline_weekday'],df['created_at_weekday'],df['launched_at_weekday']]
for i in weeks:
    i.replace(['Friday','Saturday','Sunday'], 'Weekend', inplace = True)
    i.replace(['Monday','Tuesday','Wednesday','Thursday'], 'Weekday', inplace = True)

#condense countries into continents/larger regions 
df['country'].replace(['NZ'], 'AU', inplace = True)
df['country'].replace(['DE','NL','FR','IT','ES','DK','SE','IE','NO','AT','BE','LU'], 'EU', inplace = True)
df['country'].replace(['CH','HK','SG'], 'ASIA', inplace = True)

#since US is the dominant class, and this variable is highly correlated with country, condense into US vs. non US
df['currency'].replace(['GBP','EUR','CAD','AUD','DKK','NZD','SEK','CHF','NOK','MXN','HKD','SGD'],'Other', inplace = True)

########################## PART 2: REGRESSION ##########################

#create a copy of the df for the regression analysis
df_reg = df.copy(deep=True)

#2.1: DUMMY ENCODING & CREATION OF X, Y VARIABLES________________________________________________

#drop state variable which is only used for classification and not regression
df_reg.drop(['state'], axis = 1, inplace = True)

#dummy encode remaining predictors and create X and y variables
X_reg = pd.get_dummies(df_reg, columns = ['country','currency','category','deadline_weekday', 'created_at_weekday','launched_at_weekday','deadline_month','created_at_month','launched_at_month','launched_at_hr','launched_at_yr'])
y_reg = df_reg['usd_pledged']
X_reg.drop(['usd_pledged'],axis =1, inplace = True)

X_reg.drop(['launched_at_yr_2017','launched_at_yr_2016','deadline_month_3','created_at_month_1','created_at_month_3','launched_at_month_1','country_MX','country_GB','country_EU','deadline_weekday_Weekend','country_CA','country_ASIA','launched_at_month_2','launched_at_month_3','launched_at_hr_9','launched_at_yr_2009','launched_at_yr_2010','launched_at_yr_2011','launched_at_yr_2015','country_AU','deadline_weekday_Weekday','deadline_month_2', 'created_at_weekday_Weekday','category_Gadgets','currency_Other','created_at_month_2','deadline_month_1','launched_at_weekday_Weekend','created_at_weekday_Weekend', 'country_US','launched_at_yr_2012','launched_at_hr_4','static_usd_rate','blurb_len_clean','launched_at_yr_2014','launched_at_yr_2013','launched_at_hr_1'], axis = 1, inplace = True)

#2.2: OUTLIER REMOVAL______________________________________________________________

#use isolation forest to remove most extreme 2.5% of outliers from the analysis, which were visually identified during data exploration
from sklearn.ensemble import IsolationForest

iforest = IsolationForest(n_estimators = 100,random_state=10, contamination=0.025) 

#predict anomalies
pred = iforest.fit_predict(X_reg)
#anomaly score of each 
iforest_scores = iforest.decision_function(X_reg)

#create a new variable to store the index number where anomaly =-1 in the anomalies vector
from numpy import where
anomaly_index = where(pred ==-1)
#extract values corresponding to the index of anomalies from the main df
anomaly_values = X_reg.iloc[anomaly_index]

#remove anomalies
X_reg.drop(index = anomaly_values.index, inplace = True)
y_reg.drop(index = anomaly_values.index, inplace = True)


#2.3: FINAL REGRESSION MODEL_____________________________________________________________

#standardize X variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_reg_std = scaler.fit_transform(X_reg)

#split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_reg_std, y_reg, test_size = 0.33, random_state = 10)

#this is the final regression model used, after comparing other models, performing hyperparameter tuning, and cross validation
from sklearn.ensemble import GradientBoostingRegressor

GBT_regression = GradientBoostingRegressor(max_depth = 6, min_samples_split = 300, min_samples_leaf =150, n_estimators = 160, random_state = 10) 

from sklearn.model_selection import cross_val_score 
neg_mse = cross_val_score(estimator = GBT_regression, X=X_reg_std, y=y_reg, scoring = 'neg_mean_squared_error', cv=5)
print('REGRESSION MODEL MSE: ', -np.average(neg_mse))

########################## PART 3: CLASSIFICATION ##########################

#create new df for classification only 
df_class = df.copy(deep=True)

#drop usd_pledged from regression which cannot be used for classification
df_class.drop(['usd_pledged'],axis=1, inplace = True)


#3.1: DUMMY ENCODING & CREATION OF X, Y VARIABLES________________________________________________

X_class = pd.get_dummies(df_class, columns = ['state','country', 'currency','category','deadline_weekday', 'created_at_weekday','launched_at_weekday','deadline_month','created_at_month','launched_at_month','launched_at_hr','launched_at_yr'])
#encode the target variable as state_successful
y_class = X_class['state_successful']

#3.2: OUTLIER REMOVAL________________________________________________


#use isolation forest to remove most extreme 2.5% of outliers from the analysis, which were visually identified during data exploration
from sklearn.ensemble import IsolationForest

iforest = IsolationForest(n_estimators = 100,random_state=10, contamination=0.025) 

#predict anomalies
pred = iforest.fit_predict(X_class)
#anomaly score of each 
iforest_scores = iforest.decision_function(X_class)

#create a new variable to store the index number where anomaly =-1 in the anomalies vector
from numpy import where
anomaly_index = where(pred ==-1)
#extract values corresponding to the index of anomalies from the main df
anomaly_values = X_class.iloc[anomaly_index]

#remove anomalies
X_class.drop(index = anomaly_values.index, inplace = True)
y_class.drop(index = anomaly_values.index, inplace = True)

#drop state from the predictors 
X_class.drop(['state_successful','state_failed'], axis =1, inplace = True)



#3.3: FINAL CLASSIFICATION MODEL_____________________________________________________________

#standardize X variables 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_class_std = scaler.fit_transform(X_class)

#split data into training and test sets                                                                                                                                 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_class_std, y_class, test_size = 0.33, random_state = 10)

#this is the final classification model used, after comparing other models, performing hyperparameter tuning, and cross validation
from sklearn.ensemble import GradientBoostingClassifier
GBT_classification = GradientBoostingClassifier(max_depth = 8, min_samples_split=800,min_samples_leaf =300, n_estimators = 100, random_state = 10)

from sklearn.model_selection import cross_val_score 
accuracy= cross_val_score(estimator = GBT_classification, X=X_class_std, y=y_class, scoring = 'accuracy', cv=5)
print('CLASSIFICATION MODEL ACCURACY: ', np.average(accuracy))

