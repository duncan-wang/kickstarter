#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:04:21 2020

@author: duncanwang
"""

#INSY 662 -- Individual Project - Full Code
#Duncan Wang -- 260710229
#11/29/2020

########################## PART 1: DATA PREPROCESSING ######################

#import packages 
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display

#read file 
df = pd.read_excel('Kickstarter.xlsx')
df.describe()


#1.1 : HANDLE NULL VALUES_____________________________________________

#null values appear in columns "category",  "launch_to_state_change_days", "blurb len", "name len" (clean and not clean)
#print(df.isnull().any())

#print(df['category'].isnull().value_counts())
#print(df['name_len'].isnull().value_counts())
#print(df['name_len_clean'].isnull().value_counts())
#print(df['blurb_len'].isnull().value_counts())
#print(df['blurb_len_clean'].isnull().value_counts())
#print(df['launch_to_state_change_days'].isnull().value_counts())

#category column has many null values -- will impute since this may be a valuable category, fill null category values with new 'Other' values 
df['category'].fillna('Other',inplace = True)
#average imputation for name and blurb lengths, since there are only 5 values 
df['name_len'].fillna(df['name_len'].mean(),inplace = True)
df['name_len_clean'].fillna(df['name_len_clean'].mean(),inplace = True)
df['blurb_len'].fillna(df['blurb_len'].mean(),inplace = True)
df['blurb_len_clean'].fillna(df['blurb_len_clean'].mean(),inplace = True)
#this column has 66% null values, so will drop
df.drop(['launch_to_state_change_days'], axis = 1, inplace = True)

#no more null values 
#print(df.isnull().any())


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

#correlation matrix
correlation_matrix = df.corr()
#sns.heatmap(correlation_matrix)

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


#from the gradient boosting feature selection process used below, drop the variables here -- this is iterative, so the gradient boost is run many times to see how many dropped predictors results in the best MSE
#drop them before running outlier detection so that the outlier detection is not influenced by the variables to be dropped
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


#2.3: FEATURE SELECTION____________________________________________________________

#standardize X variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_reg_std = scaler.fit_transform(X_reg)

#use Gradient Boost Regressor to select most important features
from sklearn.ensemble import GradientBoostingRegressor
GBT = GradientBoostingRegressor(random_state = 10)
gbt_model = GBT.fit(X_reg_std, y_reg)
feat_importances = pd.DataFrame(list(zip(X_reg.columns, gbt_model.feature_importances_)), columns = ['Predictor', 'GBT Feat. Importances']).sort_values(by = 'GBT Feat. Importances')

#2.4: MODEL COMPARISON_____________________________________________________________

#split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_reg_std, y_reg, test_size = 0.33, random_state = 10)

#Many models were tested for relative performance before deciding on the final model 

#LASSO REGRESSOR - ELIMINATED
#from sklearn.linear_model import Lasso
#lasso = Lasso(alpha = 1) 
#model = lasso.fit(X_train, y_train)
#y_test_pred = model.predict(X_test)
#from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score 
#print(mean_squared_error(y_test, y_test_pred))
#print(explained_variance_score(y_test, y_test_pred))
#print(r2_score(y_test, y_test_pred))

#KNN REGRESSOR - ELIMINATED
#from sklearn.neighbors import KNeighborsRegressor
#knn = KNeighborsRegressor(n_neighbors = 400)
#model = knn.fit(X_train, y_train)
#y_test_pred = model.predict(X_test)
#from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score 
#print(mean_squared_error(y_test, y_test_pred))
#print(explained_variance_score(y_test, y_test_pred))
#print(r2_score(y_test, y_test_pred))

#DECISION TREE REGRESSOR - ELIMINATED
#from sklearn.tree import DecisionTreeRegressor
#cart = DecisionTreeRegressor(max_depth = 2)
#model = cart.fit(X_train, y_train)
#y_test_pred = model.predict(X_test)
#from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score 
#print(mean_squared_error(y_test, y_test_pred))
#print(explained_variance_score(y_test, y_test_pred))
#print(r2_score(y_test, y_test_pred))

#MLP REGRESSOR - ELIMINATED
#from sklearn.neural_network import MLPRegressor
#ANN = MLPRegressor(hidden_layer_sizes=(5), max_iter = 1000)
#model = ANN.fit(X_train, y_train)
#y_test_pred = model.predict(X_test)
#from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score 
#print(mean_squared_error(y_test, y_test_pred))
#print(explained_variance_score(y_test, y_test_pred))
#print(r2_score(y_test, y_test_pred))


#2.5: FINAL REGRESSION MODEL_____________________________________________________________

#GridSearchCV was utilized to perform hyperparameter tuning on the regression model to find optimal level of hyperparameters  

#from sklearn.ensemble import GradientBoostingRegressor
#GBT = GradientBoostingRegressor(random_state = 10, n_estimators = 160)
#parameters = {'max_depth': [i for i in range(4,14,2)],
              #'min_samples_split': [i for i in range(100,1000,100)],
              #'min_samples_leaf': [i for i in range(100,500,100)]
              #} 

#from sklearn.model_selection import GridSearchCV
#grid_search = GridSearchCV(estimator = GBT,
                           #param_grid = parameters,
                           #cv = 5,
                           #n_jobs = -1,
                           #verbose = 2)

#grid_search = grid_search.fit(X_reg_std, y_reg)

#print("Best parameters: ",grid_search.best_params_)
#print("Best score: ",grid_search.best_score_)

#min samples leaf = min samples for a leaf node (external node) -- without children, used to control overfitting
#min samples split = min samples required to split an internal node
#max depth = max depth to grow a tree 

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


#3.3: FEATURE SELECTION_________________________________________________________

#standardize X variables 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_class_std = scaler.fit_transform(X_class)

#feature selection using gradient boost -- no variables were ultimately eliminated using this process as it was found that it did not improve the analysis
from sklearn.ensemble import GradientBoostingClassifier
GBT = GradientBoostingClassifier(random_state = 10)
gbt_model = GBT.fit(X_class_std, y_class)
pd.DataFrame(list(zip(X_class.columns, gbt_model.feature_importances_)), columns = ['Predictor', 'GBT Feat. Importances']).sort_values(by = 'GBT Feat. Importances')

#3.4: MODEL COMPARISON_____________________________________________________________

#split data into training and test sets                                                                                                                                 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_class_std, y_class, test_size = 0.33, random_state = 10)

#LOGISTIC REGRESSION - ELIMINATED
#from sklearn.linear_model import LogisticRegression
#logreg = LogisticRegression()
#model = logreg.fit(X_train,y_train)
#y_test_pred = model.predict(X_test)
#from sklearn.metrics import classification_report
#print("\nClassification report:")
#print(classification_report(y_test, y_test_pred))
#from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
#print('accuracy: ', accuracy_score(y_test, y_test_pred))
#print('precision: ', precision_score(y_test, y_test_pred))
#print('recall: ', recall_score(y_test, y_test_pred))
#print(confusion_matrix(y_test, y_test_pred))

#KNN CLASSIFIER -- ELIMINATED
#from sklearn.neighbors import KNeighborsClassifier
#for i in range(1,11):
    #knn = KNeighborsClassifier(n_neighbors = i, p = 2)
    #model = knn.fit(X_train, y_train)
    #y_test_pred = model.predict(X_test)
    #print('K Neighbors =', i, ', Accuracy Score: ',  accuracy_score(y_test, y_test_pred))
#knn = KNeighborsClassifier(n_neighbors = 8)
#model = knn.fit(X_train, y_train)
#y_test_pred = model.predict(X_test)
#print(classification_report(y_test, y_test_pred))
#print('accuracy: ', accuracy_score(y_test, y_test_pred))
#print('precision: ', precision_score(y_test, y_test_pred))
#print('recall: ', recall_score(y_test, y_test_pred))
#print(confusion_matrix(y_test, y_test_pred))

#MLP CLASSIFIER - ELIMINATED
#from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(hidden_layer_sizes=(4), max_iter = 1000, random_state =0)
#model = mlp.fit(X_train, y_train)
#y_test_pred = model.predict(X_test)
#print(classification_report(y_test, y_test_pred))
#print('accuracy: ', accuracy_score(y_test, y_test_pred))
#print('precision: ', precision_score(y_test, y_test_pred))
#print('recall: ', recall_score(y_test, y_test_pred))
#print(confusion_matrix(y_test, y_test_pred))

#DECISION TREE/CART - ELIMINATED
#from sklearn.tree import DecisionTreeClassifier
#decisiontree = DecisionTreeClassifier(max_depth = 5)
#model = decisiontree.fit(X_train, y_train)
#y_test_pred = model.predict(X_test)
#print(classification_report(y_test, y_test_pred))
#print('accuracy: ', accuracy_score(y_test, y_test_pred))
#print('precision: ', precision_score(y_test, y_test_pred))
#print('recall: ', recall_score(y_test, y_test_pred))
#print(confusion_matrix(y_test, y_test_pred))

#RANDOM FOREST - ELIMINATED
#from sklearn.ensemble import RandomForestClassifier 
#randomforest = RandomForestClassifier(random_state = 0, max_features = 11, n_estimators = 100)
#model = randomforest.fit(X_train, y_train)
#y_test_pred = model.predict(X_test)
#print(classification_report(y_test, y_test_pred))
#print('accuracy: ', accuracy_score(y_test, y_test_pred))
#print('precision: ', precision_score(y_test, y_test_pred))
#print('recall: ', recall_score(y_test, y_test_pred))
#print(confusion_matrix(y_test, y_test_pred))


#3.5: FINAL CLASSIFICATION MODEL_____________________________________________________________

#GridSearchCV was utilized to perform hyperparameter tuning on the regression model to find optimal level of hyperparameters  

#GBT = GradientBoostingClassifier(random_state = 10, n_estimators = 100)
#parameters = {'max_depth': [i for i in range(4,14,2)],
              #'min_samples_split': [i for i in range(100,1000,100)],
              #'min_samples_leaf': [i for i in range(100,500,100)]
              #} 

#from sklearn.model_selection import GridSearchCV
#grid_search = GridSearchCV(estimator = GBT,
                           #param_grid = parameters,
                           #cv = 5,
                           #n_jobs = -1,
                           #verbose = 2)

#grid_search = grid_search.fit(X_std, y)

#print("Best parameters: ",grid_search.best_params_)
#print("Best score: ",grid_search.best_score_)

#min samples leaf = min samples for a leaf node (external node) -- without children, used to control overfitting
#min samples split = min samples required to split an internal node
#max depth = max depth to grow a tree 

#this is the final classification model used, after comparing other models, performing hyperparameter tuning, and cross validation
from sklearn.ensemble import GradientBoostingClassifier
GBT_classification = GradientBoostingClassifier(max_depth = 8, min_samples_split=800,min_samples_leaf =300, n_estimators = 100, random_state = 10)

from sklearn.model_selection import cross_val_score 
accuracy= cross_val_score(estimator = GBT_classification, X=X_class_std, y=y_class, scoring = 'accuracy', cv=5)
print('CLASSIFICATION MODEL ACCURACY: ', np.average(accuracy))


########################## PART 4: CLUSTERING ##########################


#create a copy of the df for clustering only
df_cluster = df.copy(deep=True)

#choose 3 variables that were most important based on the feature importances scores from regression and classification
X_cluster = df_cluster[['goal_usd', 'launch_to_deadline_days','name_len_clean']]

#4.1: OUTLIER REMOVAL_____________________________________________________________

#since cluster formation is sensitive to outliers, use anomaly detection to remove 2.5% of extreme outliers

from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators = 100,random_state=10, contamination=0.025) 

#predict anomalies
pred = iforest.fit_predict(X_cluster)
#anomaly score of each 
iforest_scores = iforest.decision_function(X_cluster)

#create a new variable to store the index number where anomaly =-1 in the anomalies vector
from numpy import where
anomaly_index = where(pred ==-1)
#extract values corresponding to the index of anomalies from the main df
anomaly_values = X_cluster.iloc[anomaly_index]

#remove anomalies
X_cluster.drop(index = anomaly_values.index, inplace = True)

#4.2: DETERMINE OPTIMAL NUMBER OF CLUSTERS_____________________________________________________________

#standardize dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_cluster_std = scaler.fit_transform(X_cluster)

#find optimal number of clusters which achieves the greatest reduction in inertia (within cluster variance)
from sklearn.cluster import KMeans
within_ss = []
for i in range(2,8):
    kmeans = KMeans(n_clusters = i)
    model = kmeans.fit(X_cluster_std)
    within_ss.append(model.inertia_)
    print('clusters: ', i, ', inertia: ', model.inertia_)
    
#plot elbow plot -- within cluster variance is consistently reduced up to 4 clusters, and then plateaus
from matplotlib import pyplot as plt 
plt.plot([2,3,4,5,6,7], within_ss)


#4.3: K-MEANS CLUSTERING MODEL_____________________________________________________________

#create K-Means clustering model with 4 clusters as determined by the model inertia
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, random_state = 10)
model = kmeans.fit(X_cluster_std)
#model labels 
labels = model.labels_ 
#labels of clusters assigned
model.cluster_centers_

#calculate silhouette score (between and within cluster variation) 
#silhoutette score interpretation: 0.5+ = good, 0.25-0.5 = okay, <0.25 = bad  
from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X_cluster_std,labels)

#silhouette score of individual clusters
df_s = pd.DataFrame({'label': labels, 'silhouette': silhouette})
print('Avg Silhouette Score for Cluster 1: ', np.average(df_s[df_s['label'] == 0].silhouette))
print('Avg Silhouette Score for Cluster 2: ', np.average(df_s[df_s['label'] == 1].silhouette))
print('Avg Silhouette Score for Cluster 3: ', np.average(df_s[df_s['label'] == 2].silhouette))
print('Avg Silhouette Score for Cluster 4: ', np.average(df_s[df_s['label'] == 3].silhouette))

#total silhouette score
from sklearn.metrics import silhouette_score
print('OVERALL SILHOUETTE SCORE FOR 4 CLUSTERS: ' , silhouette_score(X_cluster_std,labels))


#4.4: CLUSTER RESULT VISUALIZATIONS____________________________________________________

#add cluster labels to dataframe 
X_cluster.loc[:,'cluster_labels'] = labels
X_cluster.loc[:,'state'] = df_cluster.loc[:,'state']

#create new df for visualizations
X_qt = X_cluster.copy(deep=True)
X_qt.reset_index(inplace = True)

#calculate percentiles for each of the observations in each group to group them from 0-25, 26-50, 51-75, and 76-100th percentiles
cols = ['goal_usd','launch_to_deadline_days','name_len_clean']
new_cols = ['goal_usd1','launch_to_deadline_days1','name_len_clean1']
for i in new_cols:
    X_qt[i] = 0
for k in range(len(cols)):
    percentile = np.percentile(X_qt[cols[k]], [25,50,75])
    for i in range(len(X_qt)):
        if X_qt[cols[k]][i] <= percentile[0]:
            X_qt[new_cols[k]][i] = '0-25'
        elif X_qt[cols[k]][i] <= percentile[1]:
            X_qt[new_cols[k]][i] = '26-50'
        elif X_qt[cols[k]][i] <= percentile[2]:
            X_qt[new_cols[k]][i] = '51-75'
        else:
            X_qt[new_cols[k]][i] = '76-100'
            
#rename clusters
X_qt.loc[X_qt['cluster_labels'] ==0, 'cluster_labels'] = 'Cluster 1'
X_qt.loc[X_qt['cluster_labels'] ==1, 'cluster_labels'] = 'Cluster 2'
X_qt.loc[X_qt['cluster_labels'] ==2, 'cluster_labels'] = 'Cluster 3'
X_qt.loc[X_qt['cluster_labels'] ==3, 'cluster_labels'] = 'Cluster 4'

import plotly.express as px

#CLUSTER DISTRIBUTION FOR GOAL USD 
fig = px.histogram(X_qt, x = 'cluster_labels', color= 'goal_usd1',
                   labels=dict(goal_usd1 = 'Percentile'), 
                   category_orders={'goal_usd1': ['0-25', '26-50', '51-75', '75-100']}).update_xaxes(categoryorder='category ascending')

fig.update_layout(title_text='Goal (USD)', title_x=0.5)
fig.show()

#CLUSTER DISTRIBUTION FOR LAUNCH TO DEADLINE (DAYS)
fig = px.histogram(X_qt, x = 'cluster_labels', color= 'launch_to_deadline_days1',
                   labels=dict(launch_to_deadline_days1 = 'Percentile'), 
                   category_orders={'launch_to_deadline_days': ['0-25', '26-50', '51-75', '75-100']}).update_xaxes(categoryorder='category ascending')

fig.update_layout(title_text='Launch to Deadline (Days)', title_x=0.5)
fig.show()

#CLUSTER DISTRIBUTION FOR NAME LENGTH - CLEAN
fig = px.histogram(X_qt, x = 'cluster_labels', color= 'name_len_clean1',
                   labels=dict(name_len_clean1 = 'Percentile'), 
                   category_orders={'name_len_clean1': ['0-25', '26-50', '51-75', '75-100']}).update_xaxes(categoryorder='category ascending')

fig.update_layout(title_text='Name Length (Key Words)', title_x=0.5)
fig.show()

#CLUSTER DISTRIBUTION FOR NAME For STATE - FAIL/SUCCESS
fig = px.histogram(X_qt, x = 'cluster_labels', color= 'state',
                   labels=dict(state = 'Percentile'), 
                   category_orders={'state': ['0-25', '26-50', '51-75', '75-100']},color_discrete_sequence=["cornflowerblue","yellow"]).update_xaxes(categoryorder='category ascending')

fig.update_layout(title_text='Project State', title_x=0.5)
fig.show()
