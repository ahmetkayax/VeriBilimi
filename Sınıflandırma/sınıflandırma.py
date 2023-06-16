# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:18:39 2023

@author: AhmetKaya
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score       
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error,r2_score, roc_auc_score, roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

#Veri Seti Hikayesi ve Problem: Şeker Hastalığı Tahmini ¶|
df = pd.read_csv("./diabetes.csv")
df.head()

#Lojistik Regresyon Model & Tahmin

df["Outcome"].value_counts()
df.describe().T

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
y.head()
X.head()

loj_model=LogisticRegression(solver="liblinear").fit(X,y)
loj_model.intercept_
loj_model.coef_
loj_model.predict(X)[0:10]
y[0:10]
y_pred=loj_model.predict(X)
confusion_matrix(y, y_pred)
accuracy_score((y), y_pred)
print(classification_report(y, y_pred))
loj_model.predict_proba(X)[0:10]


#ROG EĞRİSİ
logit_roc_auc = roc_auc_score(y, loj_model.predict(X))
fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X) [:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim( [0.0, 1.0])
plt.ylim( [0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt. legend (loc="lower right")
plt.savefig('Log_ROC')
plt.show()


#Model Tuning (Model Doğrulama)
X_train, X_test, y_train, y_test = train_test_split(X,y,
test_size=0.30, random_state=42)
loj_model = LogisticRegression (solver = "liblinear").fit(X_train, y_train)
y_pred = loj_model.predict(X_test)
print(accuracy_score (y_test, y_pred))

cross_val_score(loj_model,X_test,y_test,cv=10).mean()



#K-EN YAKIN KOMŞU (KNN)

df.head()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# K-EN YAKIN KOMŞU (KNN) MODEL & TAHMİN
df.head()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

knn_model = KNeighborsClassifier().fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


#K-EN YAKIN KOMŞU (KNN) MODEL TUNING

knn=KNeighborsClassifier()
knn_params = {"n_neighbors": np.arange(1,50)}
knn_cv_model = GridSearchCV(knn,knn_params, cv=10 ).fit(X_train,y_train)
knn_cv_model.best_score_
knn_cv_model.best_params_

#Final Model

knn_tuned = KNeighborsClassifier(n_neighbors=11).fit(X_train, y_train)
y_pred= knn_tuned.predict(X_test)
accuracy_score(y_test,y_pred)
knn_tuned.score(X_test,y_test)


#DESTEK VEKTÖR MAKİNELERİ (SVM)
df.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=42)                                        



#DESTEK VEKTÖR MAKİNELERİ (SVM) MODEL & TAHMIN

svm_model = SVC(kernel="linear").fit(X_train,y_train)
svm_model
y_pred = svm_model.predict(X_test)
accuracy_score(y_test,y_pred)


#DESTEK VEKTÖR MAKİNELERİ (SVM) MODEL TUNING

svm = SVC()
svm_params = {"C": np.arange(1, 10), "kernel": ["linear", "rbf"]}
svm_cv_model = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1, verbose=2)
svm_cv_model.fit(X_train, y_train)

svm_cv_model.best_score_
svm_cv_model.best_params_

#Final Model

svm_tuned = SVC(C=2,kernel="linear").fit(X_train,y_train)
y_pred=svm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


#YAPAY SİNİR AĞLARI(ÇOK KATMANLI ALGILAYICILAR)
df.head()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=42)

scaler= StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
scaler.fit(X_test)
X_train= scaler.transform(X_test)

#MODEL & TAHMIN

mlpc_model = MLPClassifier().fit(X_train, y_train)
mlpc_model.coefs_
y_pred = mlpc_model.predict(X_test)
accuracy_score(y_test,y_pred)


#MODEL TUNING

mlpc_params = {"alpha":[1,5,0.1,0.01,0.03,0.005,0.0001],
               "hidden_layer_sizes":[(10,10),(100,100,100),(100,100),(3,5)]}
mlpc = MLPClassifier(solver="lbfgs",activation="logistic")
mlpc_cv_model = GridSearchCV(mlpc,mlpc_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)

mlpc_cv_model
mlpc_cv_model.best_params_

#Final Model

mlpc_tuned = MLPClassifier(solver="lbfgs",activation='logistic' ,alpha=5, hidden_layer_sizes=(3, 5)).fit(X_train, y_train)
y_pred = mlpc_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


#CART (Classification and Regression Tree)
df.head()

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.30,
                                                 random_state=42)

#CART MODEL & TAHMIN

cart_model = DecisionTreeClassifier().fit(X_train,y_train)
cart_model
y_pred=cart_model.predict(X_test)
accuracy_score(y_test, y_pred)

#CART MODEL TUNING

cart =DecisionTreeClassifier()
cart_params ={"max_depth": [1,3,5,8,10],
              "min_samples_split":[2,3,5,10,20,50]}
cart_cv_model=GridSearchCV(cart,cart_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
cart_cv_model.best_params_

#FINAL MODEL

cart_tuned=DecisionTreeClassifier(max_depth=5,min_samples_split=20).fit(X_train,y_train)
y_pred=cart_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

#RANDOM FORESTS
df.head()
X_trian,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.30,
                                               random_state=42)

#RANDOM FOREST MODEL & TAHMIN

rf_model= RandomForestClassifier().fit(X_train,y_train)
rf_model
y_pred= rf_model.predict(X_test)
accuracy_score(y_test, y_pred)

#RANDOM FOREST MODEL TUNING
X_train.shape
rf= RandomForestClassifier()
rf_params = {"n_estimators": [100, 200, 500, 1000],
             "max_features": [3, 5, 7],
             "min_samples_split": [2, 5, 10, 20]}
rf_cv_model = GridSearchCV(rf, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
rf_cv_model.best_params_
rf_tuned=RandomForestClassifier(max_features=8,
                                min_samples_split=5,
                                n_estimators=500).fit(X_train,y_train)
y_pred=rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


#Değişken Önem Düzeyleri

rf_tuned
feature_imp = pd.Series (rf_tuned.feature_importances_,
                         index=X_train.columns).sort_values (ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()


#GRADIENT BOOSTİNG MACHİNES
df.head()

#GRADIENT BOOSTİNG MACHİNES MODEL & TAHMIN

gbm_model = GradientBoostingClassifier().fit(X_train,y_train)
gbm_model

y_pred= gbm_model.predict(X_test)
accuracy_score(y_test, y_pred)

#GRADIENT BOOSTİNG MACHİNES MODEL TUNING

gbm= GradientBoostingClassifier()

gbm_params ={"learning_rate":[0.1,0.001,0.05],
             "n_estimators":[100,300,500,100],
             "max_depth":[2,3,5,8]}

gbm_cv_model = GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
gbm_cv_model.best_params_
gbm_tuned = GradientBoostingClassifier(learning_rate=0.01,
                                       max_depth=5,
                                       n_estimators=500).fit(X_train, y_train)

y_pred=gbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


#Değişken Önem Düzeyleri

gbm_tuned
feature_imp = pd.Series (gbm_tuned.feature_importances_,
                         index=X_train.columns).sort_values (ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()



""" HOCAM BURADA HATA VERDİ ÇÖZEMEDİM HATAYI O YÜZDEN BU KISMI KAPATTIM.."""

"""#XGBOOST
from xgboost import XGBClassifier
xgb_model = XGBClassifier().fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred)"""


"""
#LIGHT GBM
df.head()


#LIGHT GBM MODEL & TAHMIN

lgbm_model = LGBMClassifier().fit(X_train, y_train)
y_pred =lgbm_model.predict(X_test)
accuracy_score(y_test, y_pred)


#LIGHT GBM MODEL TUNING

lgbm = LGBMClassifier()
lgbm_params = {"learning_rate":[0.001,0.01,0.1],
               "n_estimators":[200,500,100],
               "max_depth":[1,2,35,8]}
lgbm_cv_model = GridSearchCV(lgbm,lgbm_params,cv=10,n_jobs=-1,verbose=2).fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned  = LGBMClassifier(learning_rate= 0.01,
                                   max_depth= 1,
                                   n_estimators= 500).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

#Değişken Önem Düzeyleri

lgbm_tuned
feature_imp = pd.Series (lgbm_tuned.feature_importances_,
                         index=X_train.columns).sort_values (ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show() """











"""
#CATBOOST
df.head()

#CATBOOST MODEL & TAHMİN

catb_model = CatBoostClassifier().fit(X_train,y_train,verbose=False)
y_pred=catb_model.predict(X_test)
accuracy_score(y_test, y_pred)

#CATBOOST MODEL TUNING

catb= CatBoostClassifier(verbose=False)
catb_params={"iterations":[200,500,1000],
             "learning_rate":[0.01,0.03,0.1],
             "dept":[4,5,8]}
catb_cv_model = GridSearchCV(catb,catb_params,
                             cv=5,n_jobs=-1,verbose=2).fit(X_train,y_train)
catb_cv_model.best_params_
catb_tuned = CatBoostClassifier(depth=8,iterations=200,learning_rate=0.03).fit(X_train,y_train)
y_pred=catb_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

#Değişken Önem Düzeyleri

catb_tuned
feature_imp = pd.Series (catb_tuned.feature_importances_,
                         index=X_train.columns).sort_values (ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()"""










