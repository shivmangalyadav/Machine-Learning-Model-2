# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:57:19 2019

@author: lucky yadav
"""
import os
#============================================================================
#Predicting Price Of Pre-owned car
#===========================================================================
import pandas as pd
import numpy as np
import seaborn as sns
os.chdir("D:\python programs")
#==========================================================================
#setting the dimension for the plot
#============================================================================
sns.set(rc={"figure.figsize":(11.7,8.27)})


#==============================================================================================
#Reading the file
#=============================================================================================
cars_data=pd.read_csv("cars_sampled.csv")
cars=cars_data.copy()
#==================================================================================
#structre of the data
#=======================================================================================

cars.info()

#===============================================================================
#sumarizing data
#==================================================================================
cars.describe()
pd.set_option("display.float_format",lambda x:"%.3f" %x)
cars.describe()

#To Display Maximum columns
#==========================================================================================
pd.set_option("display.max_columns",500)
cars.describe()
#=======================================================================================
#Dropping unwanted columns
#=========================================================================================
cols=["name","dateCreated","postalCode","lastSeen","dateCrawled"]
cars=cars.drop(columns=cols,axis=1)
#removing Duplicate records
cars.drop_duplicates(keep="first",inplace=True) 
# 470 duplicates 
#==============================================================================================
# data cleaning
#=============================================================================================
 

#number of missing values in each column
 
cars.isnull().sum()
# year of registration
year_counts= cars["yearOfRegistration"].value_counts().sort_index()
sum(cars["yearOfRegistration"]>2018)
sum(cars["yearOfRegistration"]<1950)
sns.regplot(x="yearOfRegistration",y="price",scatter=True,fit_reg=False,data=cars) 
#working Range 1950 to 2018
price_counts=cars["price"].value_counts().sort_index()
sns.distplot(cars["price"])
cars["price"].describe()
sns.boxplot(y=cars["price"])
sum(cars["price"]>15000)
sum(cars["price"]<100)
#working range 100 to 150000

#variable PowerPs
power_counts=cars["powerPS"].value_counts().sort_index()
sns.distplot(cars["powerPS"])
cars["powerPS"].describe()
sns.boxplot(y=cars["powerPS"])
sns.regplot(x=cars["powerPS"],y=cars["price"],scatter=True,fit_reg=False,data=cars)
sum(cars["powerPS"]>500)
sum(cars["powerPS"]<10)
#working range 10 to 500
#====================================================================================
#working of data
#======================================================================================
cars=cars[(cars.yearOfRegistration <=2018)
& (cars.yearOfRegistration >= 1950)
& (cars.price >=100)
& (cars.price <=150000)
& (cars.powerPS >=10)
& (cars.powerPS <=500)]
# 6700 records are dropped

#Further to simplify variable  reduction
#Combining yearOfregistration month of registraton 
cars["monthOfRegistration"]/=12

#creating a new variable age by adding yearOfRegistration and month  of Registraion 
cars["age"]=(2018-cars["yearOfRegistration"])+cars["monthOfRegistration"]
cars["age"]=round(cars["age"],2)
cars["age"].describe()
cars=cars.drop(columns=["yearOfRegistration","monthOfRegistration"],axis=1)

#Visalizing of Paramater
#age
sns.distplot(cars["age"])
sns.boxplot(y=cars["age"])

#price

sns.distplot(cars["price"])
sns.boxplot(y=cars["price"])

#Powerps

sns.distplot(cars["powerPS"])
sns.boxplot(y=cars["powerPS"])

#visuallizing the parameter after the narrowing range
sns.regplot(x="age",y="price",scatter=True,fit_reg=False,data=cars)
#cars price are higher and newer
#as age is increases price is decreases 
#However some cars are higher price as  is increases


#==============================================================================
#power vs price
sns.regplot(x="powerPS",y="price",scatter=True,fit_reg=False,data=cars)

#variable seller
cars["seller"].value_counts()
pd.crosstab(cars["seller"],columns="count",normalize=True)
sns.countplot(x="seller",data=cars)

#variable offer type
cars["offerType"].value_counts()
sns.countplot(x="offerType",data=cars)

#variable abtest
cars["abtest"].value_counts()
pd.crosstab(cars["abtest"],columns="count",normalize=True)
sns.countplot(x="abtest",data=cars)
sns.boxplot(x="abtest",y="price",data=cars)
# variable vehical type
cars["vehicleType"].value_counts()
pd.crosstab(cars["vehicleType"],columns="count",normalize=True)
sns.countplot(x="vehicleType",data=cars)
sns.boxplot(x="vehicleType",y="price",data=cars)
# variable  gearbox
cars["gearbox"].value_counts()
pd.crosstab(cars["gearbox"],columns="count",normalize=True)
sns.countplot(x="gearbox",data=cars)
sns.boxplot(x="gearbox",y="price",data=cars)
#variable Model
cars["model"].value_counts()
pd.crosstab(cars["model"],columns="count",normalize=True)
sns.countplot(x="model",data=cars)
sns.boxplot(x="model",y="price",data=cars)

# variable Kilomter
cars["kilometer"].value_counts().sort_index()
pd.crosstab(cars["kilometer"],columns="count",normalize=True)
sns.countplot(x="kilometer",data=cars)
sns.boxplot(x="kilometer",y="price",data=cars)
sns.regplot(x="kilometer",y="price",scatter=True,fit_reg=False,data=cars)
sns.distplot(cars["kilometer"])
# variable fuel type
cars["fuelType"].value_counts()
pd.crosstab(cars["fuelType"],columns="count",normalize=True)
sns.countplot(x="fuelType",data=cars)
sns.boxplot(x="fuelType",y="price",data=cars)
# variable brand
cars["brand"].value_counts()
pd.crosstab(cars["brand"],columns="count",normalize=True)
sns.countplot(x="brand",data=cars)
sns.boxplot(x="brand",y="price",data=cars)
# variable not repaired damage
cars["notRepairedDamage"].value_counts()
pd.crosstab(cars["notRepairedDamage"],columns="count",normalize=True)
sns.countplot(x="notRepairedDamage",data=cars)
sns.boxplot(x="notRepairedDamage",y="price",data=cars)
# removing insignificant variablecols["seller]
cols=["seller","offerType","abtest"]
cars=cars.drop(columns=cols,axis=1)
cars_copy=cars.copy()
#correlation
cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,"price"].abs().sort_values(ascending=False)[1:]
cars_omit=cars.dropna(axis=0)
cars_omit=pd.get_dummies(cars_omit,drop_first=True)


#importing necessary liraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
 #building model with omitted data
 #seprating input features
x1=cars_omit.drop(["price"],axis="columns",inplace=False)
y1=cars_omit["price"]
#plotting the variable price
price=pd.DataFrame({"1.Before":y1,"2.After":np.log(y1)})
price.hist()
#Trans forming  prices as A LOGARTHMIC VALUE
y1=np.log(y1)
#splitting data into test and train
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
 

#baseline model for omited data
base_pred=np.mean(y_test)
print(base_pred)
#repeating same values till length of test data
base_pred=np.repeat(base_pred,len(y_test))


#finding rmse
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)

#Linear with omitted data
# setting intercept as true
lgr=LinearRegression(fit_intercept=True)

#model
model_lin1=lgr.fit(x_train,y_train)
#predicting model on test set
cars_predictions_lin1=lgr.predict(x_test)
#computing mse rmse
lin_mse1=mean_squared_error(y_test,cars_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

#R squared value
r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#regressiion diagnostics residual plot  analysis
residuals1=y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1,y=residuals1,scatter=True,fit_reg=False,data=cars)
residuals1.describe()



# random forest with omited datsa

