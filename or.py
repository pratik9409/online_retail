# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:08:14 2020

@author: ADMIN
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import matplotlib

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

#load the dataset
data=pd.read_csv("E:\data_sci\onlineretail\OnlineRetail.csv",encoding='unicode_escape')

data.info()
data.describe()

#find the nan values
data.isnull().any()
data.isnull().sum()
sns.heatmap(data.isnull(), cmap='viridis')

#removing nan values
data.dropna(inplace=True)
sns.heatmap(data.isnull(),cmap='viridis')

#correlation 
data_corr=data.corr()
plt.figure(figsize = (9,6))
sns.heatmap(data_corr,annot=True,cmap='viridis')

#converting datatype
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['Month']= pd.to_datetime(data['InvoiceDate']).dt.month
data['Days'] = pd.to_datetime(data['InvoiceDate']).dt.day
data['Year'] = pd.to_datetime(data['InvoiceDate']).dt.year
data['Hour'] = pd.to_datetime(data['InvoiceDate']).dt.hour
data['YearMonth']=pd.to_datetime(data['InvoiceDate']).dt.to_period('M')

data['YearMonth']=data['YearMonth'].astype(str)


#create a new column showing a specific number with respect to each country
data['Country_Codes']=data["Country"].astype('category').cat.codes
data['InvoiceNo']=data["InvoiceNo"].astype('category').cat.codes
data['StockCode']=data["StockCode"].astype('category').cat.codes
data['Description']=data["Description"].astype('category').cat.codes

#create a dataframe showing the number represented by each country
a=data['Country'].unique()
b=data['Country_Codes'].unique()
Country_Data = a,b
Countries = pd.DataFrame(data=data)

#calculating total amount

data['TotalAmount']=data['Quantity']*data['UnitPrice']



#boxplot of Quantity
plt.figure(figsize = (10,6))
sns.boxplot(data['Quantity'],data=data)
plt.xlabel("Quantity")
plt.grid(linestyle='-.',linewidth = .5)


#removing outliers of Quantity 
max_quantity=data['Quantity'].quantile(0.90)
outl_quant=data[data['Quantity']>max_quantity]
min_quantity=data['Quantity'].quantile(0.10)
out_quanty=data[data['Quantity']<min_quantity]
data=data[(data['Quantity']<max_quantity) & (data['Quantity']>min_quantity)]

#boxplot of quantity after removing outliers
plt.figure(figsize = (8,6))
sns.boxplot(data['Quantity'],data=data)
plt.xlabel("Quantity")
plt.grid(linestyle='-.',linewidth = .5)

#boxplot of UnitPrice
plt.figure(figsize = (10,6))
sns.boxplot(data['UnitPrice'],data=data)
plt.xlabel("UnitPrice")
plt.grid(linestyle='-.',linewidth = .5)


#removing outliers of UnitPrice
max_UnitPrice=data['UnitPrice'].quantile(0.85)
out_UP=data[data['UnitPrice']>max_UnitPrice]
min_UnitPrice=data['UnitPrice'].quantile(0.15)
out_UP2=data[data['UnitPrice']<min_UnitPrice]
data=data[(data['UnitPrice']<max_UnitPrice) & (data['UnitPrice']>min_UnitPrice)]

#boxplot of UnitPrice
plt.figure(figsize = (10,6))
sns.boxplot(data['UnitPrice'],data=data)
plt.xlabel("UnitPrice")
plt.grid(linestyle='-.',linewidth = .5)



#boxplot of TotalAmount
plt.figure(figsize = (10,6))
sns.boxplot(data['TotalAmount'],data=data)
plt.xlabel("TotalAmount")
plt.grid(linestyle='-.',linewidth = .5)


#removing outliers of TotalAmount
max_TotalAmount=data['TotalAmount'].quantile(0.92)
out_TA=data[data['TotalAmount']>max_TotalAmount]
min_TotalAmount=data['TotalAmount'].quantile(0.08)
out_TA2=data[data['TotalAmount']<min_TotalAmount]
data=data[(data['TotalAmount']<max_TotalAmount) & (data['TotalAmount']>min_TotalAmount)]

#boxplot of TotalAmount
plt.figure(figsize = (10,6))
sns.boxplot(data['TotalAmount'],data=data)
plt.xlabel("TotalAmount")
plt.grid(linestyle='-.',linewidth = .5)

#####Plottings

#bargraph for monthly
plt.figure(figsize=(12,7))
sns.barplot(x='Month', y='Quantity', data=data)
plt.xlabel('Month')
plt.ylabel('Quantity')


#bargraph for country
plt.figure(figsize=(12,7))
sns.barplot(x='Country_Codes', y='Quantity', data=data)
plt.xlabel('Country')
plt.ylabel('Quantity')


#bar graph for monthyear
plt.figure(figsize=(12,7))
sns.barplot(x='YearMonth', y='Quantity', data=data)
plt.xlabel('Country')
plt.ylabel('Quantity')


#bargraph for monthly amount
plt.figure(figsize=(12,7))
sns.barplot(x='Month', y='TotalAmount', data=data)
plt.xlabel('Month')
plt.ylabel('TotalAmount')


#bargraph for countrywise totalamount
plt.figure(figsize=(12,7))
sns.barplot(x='Country', y='TotalAmount', data=data)
plt.xlabel('Country')
plt.ylabel('TotalAmount')


#bar graph for yearmonth total amount
plt.figure(figsize=(12,7))
sns.barplot(x='YearMonth', y='TotalAmount', data=data)
plt.xlabel('MonthYear')
plt.ylabel('TotalAmount')

#histogram for quantity
plt.figure(figsize=(12,7))
plt.hist(data['Quantity'],color='orange', bins=20)
plt.show()

#histogram for TotalAmount
plt.figure(figsize=(12,7))
plt.hist(data['TotalAmount'],color='orange', bins=20)
plt.show()


#histogram for Unitprice
plt.figure(figsize=(12,7))
plt.hist(data['UnitPrice'],color='orange', bins=20)
plt.show()


#distplot
plt.figure(figsize = (12,7))
sns.distplot(data['Quantity'],kde=True,bins=15)
plt.xlabel("Quantity")
plt.grid(linestyle='-.',linewidth = .5)


plt.figure(figsize = (12,7))
sns.distplot(data['TotalAmount'],kde=True,bins=15)
plt.xlabel("TotalAmount")
plt.title("Destribution of frequency")
plt.grid(linestyle='-.',linewidth = .5)

#line
plt.figure(figsize=(12,7))
sns.lineplot(x='Month',y='Quantity' , data=data)
plt.xlabel('Month')
plt.ylabel('Quantity')



#scatterplot
plt.figure(figsize=(12,7))
plt.scatter(data['Country'],data['CustomerID'],color="blue")
plt.show()

plt.scatter(data['Month'],data['Quantity'],color="blue")
plt.show()

plt.scatter(data['Month'],data['UnitPrice'],color="blue")
plt.show()

plt.scatter(data['Quantity'],data['UnitPrice'],color="blue")
plt.show()

plt.scatter(data['Month'],data['TotalAmount'],color="blue")
plt.show()

#regplot
sns.regplot(data['TotalAmount'],data['Country'],color="blue")
plt.show()

#boxplot of Quantity for monthly analysis
plt.figure(figsize=(12,7))
sns.boxplot(x='YearMonth', y = 'Quantity', data = data)
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.title("Monthly analysis of Quantity")
plt.grid(linestyle='-.',linewidth = .2)

#boxplot of TotalAmount for monthly analysis
plt.figure(figsize=(12,7))
sns.boxplot(x='YearMonth', y = 'TotalAmount', data = data)
plt.xlabel('Month')
plt.ylabel('TotalAmount')
plt.title("Monthly analysis of TotalAmount")
plt.grid(linestyle='-.',linewidth = .2)



#Independent and dependent variables
X=data.drop(['Country_Codes','YearMonth','InvoiceDate','Country'],axis=1)
Y=data['Country_Codes'].values.reshape(-1,1)

'''
Linear_reg = LinearRegression()
mse=cross_val_score(Linear_reg,X,Y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)

lasso=Lasso(alpha=0.2,)
lasso.fit(X,Y)  
lasso_coef=lasso.coef_
print(lasso_coef)

'''
#Lasso
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model,X,Y,scoring="neg_mean_squared_error",cv = 5))
    return(rmse)

model_lasso = LassoCV(alphas = [1, 0.1 , 0.001, 0.0005], selection = 'random', max_iter=15000).fit(X,Y)
res= rmse_cv(model_lasso)
print("Mean:",res.mean())
print("Min:",res.min())

coeff = pd.Series(model_lasso.coef_,index = X.columns)
print("Lasso picked"+ str(sum(coeff!=0))+"Variables and eliminated and the other"+ str(sum(coeff == 0))+" variables")


imp_coef = pd.concat([coeff.sort_values().head(10),coeff.sort_values().tail(10)])
   
matplotlib.rcParams['figure.figsize'] =(8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


#ridge
model_ridge = RidgeCV(alphas = [11,1, 0.1 , 0.001, 0.0005]).fit(X,Y)
res = rmse_cv(model_ridge)

ridge_coef=model_ridge.coef_


print("Mean:",res.mean())
print("Min:",res.min())

coeff = pd.Series(model_ridge.coef_,index =X.columns)
print("Ridge Regression picked"+ str(sum(coeff!=0))+"Variables and eliminated and the other"+ str(sum(coeff == 0))+" variables")


imp_coef = pd.concat([coeff.sort_values().head(10),coeff.sort_values().tail(10)])
   
matplotlib.rcParams['figure.figsize'] =(8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Ridge Regression Model")

data1=data.copy()
data1=data1.drop(['InvoiceDate','YearMonth','Country'],axis=1)

scaler=StandardScaler()
scaler.fit(data1)
scaled_data=scaler.transform(data1)

data.dtypes
#PCA
pca=PCA(n_components=2)
pca.fit(data1)
x_pca=pca.transform(data1)

#the dimensionality is reduced to 3
x_pca.shape

#plot the x_pca
plt.scatter(x_pca[:,0],x_pca[:,1],c=data1['CustomerID'])

X1=X.drop(['InvoiceNo','Description','UnitPrice'],axis=1)
Y1=Y


x_train,x_test,y_train,y_test=train_test_split(X1,Y1,test_size=0.3)

#perform decision tree algorithm
dt=DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred=dt.predict(x_test)

#print the accuracy score for decision tree
print(accuracy_score(y_test,y_pred))

#classification report of decision tree
cr_dt=classification_report(y_test, y_pred)

#using regplot for decision tree
sns.regplot(y_test,y_pred,order=1, ci=None, scatter_kws={'color':'black', 's':10})
plt.scatter(y_test,y_pred,s=2)

#Random Forest

rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)

#print accuracy score for random forest
print(accuracy_score(y_test,y_pred))

#classification report of random forest
cr_rfc=classification_report(y_test, y_pred)

#using regplot for random forest
sns.regplot(y_test,y_pred,order=1, ci=None, scatter_kws={'color':'black', 's':10})
plt.scatter(y_test,y_pred,s=2)

'''
#perform support vector machine algorithm
svm1 = SVC()
svm1.fit(x_train,y_train)
y_pred = svm1.predict(x_test)

#print accuracy score for support vector machine
print(accuracy_score(y_test,y_pred))

#classification report of support vector machine
cr_svm1=classification_report(y_test, y_pred)

#using regplot for support vector machine
sns.regplot(y_test,y_pred,order=1, ci=None, scatter_kws={'color':'black', 's':10})
plt.scatter(y_test,y_pred,s=2)
'''
#plot a graph to finalise cluster number by using the elbow rule
k_range=range(1,11)
sse=[]
for k in k_range:
    km=KMeans(n_clusters=k)
    km.fit(data[['CustomerID','Country_Codes']])
    sse.append(km.inertia_)

#plot a graph to finalise cluster number by using the elbow rule
plt.xlabel('k')
plt.ylabel('sum of squared error')
plt.plot(k_range,sse)

#perform KMeans clustering
km1=KMeans(n_clusters=3)
y_predicted=km1.fit_predict(data[['CustomerID','Country_Codes']])
data['cluster']=y_predicted

#create datasets with different specific cluster values
cluster1=data[data.cluster==0]
cluster2=data[data.cluster==1]
cluster3=data[data.cluster==2]

#find the centroids of each clusters
centroids=km1.cluster_centers_

#plot the cluster graph
plt.scatter(cluster1.CustomerID,cluster1['Country_Codes'],color='green')
plt.scatter(cluster2.CustomerID,cluster2['Country_Codes'],color='red')
plt.scatter(cluster3.CustomerID,cluster3['Country_Codes'],color='black')
plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],color='purple',marker='*',label='centroids')
plt.xlabel('CustomerID')
plt.ylabel('Country_Codes')
plt.legend()
plt.show()

#copy data from df into new dataset for hierarchical clustering
h_data=data[['Country_Codes','CustomerID']].copy()
h_data.info()
h_data['CustomerID']=h_data["CustomerID"].astype('int')

#using dendogram to find optimal number of clusters
dendro=sch.dendrogram(sch.linkage(h_data,method='ward'))

#perform hierarchical clustering
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(h_data)



