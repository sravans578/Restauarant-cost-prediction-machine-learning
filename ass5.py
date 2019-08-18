import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/sravans/Documents/Term 2/data/ass1/zomato.csv')
# Taking the required attributes
dataset = dataset[['name','location','rest_type','rate','cuisines','approx_cost(for two people)']]
dataset.rename(columns={'approx_cost(for two people)': 'cost'}, inplace=True)

#Replacing null values
dataset['cost'] = dataset['cost'].str.replace(',', '')
dataset['rate'] = dataset['rate'].str.replace('/5', '')
dataset['rate'] = dataset['rate'].str.replace('NEW', '2.5')
dataset['rate'] = dataset['rate'].str.replace('-', '2.5')
dataset["cost"] = pd.to_numeric(dataset["cost"])
dataset["rate"] = pd.to_numeric(dataset["rate"])

dataset['rate']=dataset['rate'].fillna(dataset['rate'].mean())
dataset['cost']=dataset['cost'].fillna(dataset['cost'].mean())

#Removing null values from location
dataset = dataset.dropna(subset=['location'])
dataset = dataset.dropna(subset=['rest_type'])

# creating a copy
dataset_dup = dataset

df =dataset_dup.drop_duplicates(subset=['name', 'location'], keep='first')




dataset=dataset.dropna(subset=['cuisines'])
dataset_cui=dataset[['cuisines']]
dataset_cuiArr=pd.unique(dataset_cui['cuisines'])

# dataset_cuiArrstr
dummies_cuisines = df.set_index('name')['cuisines'].str.get_dummies(',')
dummies_rest_type = df.set_index('name')['rest_type'].str.get_dummies(',')
dummies_cuisines

# z=dummies_cuisines.iloc[:,0:5]
df.reset_index(drop=True, inplace=True)
dummies_rest_type.reset_index(drop=True, inplace=True)
dummies_cuisines.reset_index(drop=True, inplace=True)
# df = pd.concat([df1, df2], axis=1)

merged=pd.concat([df,dummies_cuisines],axis=1) 
merged = pd.concat([merged,dummies_rest_type ],axis=1) 
merged = merged.drop(columns=['cuisines'])
merged = merged.drop(columns=['rest_type'])
clean = merged

#Removing duplicates
clean = clean.T.drop_duplicates().T

clean.shape


#Checking for duplicate columns
arrr = clean.columns
arrr
ls = []
_new = dict()
for i in arrr:
    if i in _new:
        print("duplicate:" + i)
        ls.append(i)
    else:
        _new[i] = 1

#binarization of location attribute       
cleaned =  pd.get_dummies(clean, prefix=['location'], columns=['location'])
cleaned.info()

cleaned = cleaned.drop(columns=['name'])

#Checking the cloumns
arrr = cleaned.columns
for i in arrr:
    print(i)
 

Y = cleaned.iloc[:, 1].values 
Y=Y.astype('int')
#Removing cost from feature set
cleaned = cleaned.drop(columns=['cost'])
Y.shape   
X = cleaned.iloc[:, :].values   
X.shape

# Test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

#Cross validation
from sklearn.model_selection import cross_val_score

#Linear Regression
from sklearn.linear_model import LinearRegression 
reg = LinearRegression()
model=reg.fit(X_train,y_train)
scores_LR = cross_val_score(model, X_train, y_train, cv=6)
scores_LR
scores_LR.mean()
reg.score(X_test,y_test)

#logisitic regression
from sklearn.linear_model import LogisticRegression
clfLR = LogisticRegression(multi_class='auto', solver='liblinear').fit(X_train, y_train)
scores_Log = cross_val_score(clfLR, X_train, y_train, cv=6)
scores_Log
clfLR.score(X_test,y_test)

#random Forest
from sklearn.ensemble import RandomForestRegressor
RF_R = RandomForestRegressor(max_depth=None, random_state=0,n_estimators=100)
RF_R.fit(X_train, y_train)
scores_RF = cross_val_score(RF_R, X_train, y_train, cv=6) 
scores_RF
scores_RF.mean()
RF_R.score(X_test,y_test)
importance = RF_R.feature_importances_

#Adaboost (not working)
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=50)
clf.fit(X_train,y_train)
scores_ada = cross_val_score(clf, X_train, y_train, cv=6)
scores_ada
clf.score(X_test,y_test)

#KNN
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

#Using grid search cv to find optimum number of neighbors for KNN
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model_KNN = GridSearchCV(knn, params, cv=6)
model_KNN.fit(X_train,y_train)
model_KNN.best_params_


#KNN 
knn = neighbors.KNeighborsRegressor(n_neighbors=5) # calcualted after doing GridSearchCV
knn.fit(X_train,y_train)
scores_KNN = cross_val_score(knn, X_train, y_train, cv=6)
scores_KNN
knn.score(X_test,y_test)



#Plotting Results
Results=pd.DataFrame(scor*100,columns=['Logistic_regression'])
Results=Results.assign(KNN = scores_KNN*100)
#Results=Results.assign(Linear_Regression=scores_LR*100)
Results=Results.assign(Random_Forest=scores_RF*100)
Results=Results.assign(ADAboost = scores_ada*100)
Results.xlabel="Model"
Results.ylabel="Mean Acuuracy"
Results.mean().plot(kind='bar')
Results.head()


plt.bar(scores_Log)
plt.show('bar')

#https://libraries.io/pypi/sklearn-relief
import sklearn_relief as relief
r = relief.Relief(n_features=100) #CHoosing best 100 features from feature set
my_transformed_matrix = r.fit_transform(X_train,y_train)
RF_Relief = RandomForestRegressor(max_depth=None, random_state=0,n_estimators=100)
RF_Relief.fit(my_transformed_matrix,y_train)
scores_RFi = cross_val_score(RF_Relief, X_train, y_train, cv=6) 
scores_RF
scores_RF.mean()
RF_Relief.score(my_transformed_matrix,y_train)

from sklearn.model_selection import GridSearchCV
param_grid = { 
    'n_estimators': [50,100,200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['mse', 'mae']
}
RF_R_GS = RandomForestRegressor(random_state=20)
grid_rf = GridSearchCV(estimator=RF_R_GS, param_grid=param_grid, cv= 5)
grid_result = grid_rf.fit(X_train, y_train)
grid_result.best_score_
grid_result.best_params_

