#Imported Stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report



#Reading Dataset
df = pd.read_csv('data/stars csv.csv')
print(df.head())


print(df.shape)

#Checking for null values
print(df.isnull().sum())

print(df.describe())

print(df.corr())
#sns.pairplot(df)
#plt.show()

#Correlations of each feature in dataset
corrmat = df.corr()
top_features = corrmat.index
#plt.figure(figsize = (20,20))

#g = sns.heatmap(df[top_features].corr(), annot = True, cmap = "Blues")
#plt.show()

#plt.figure()
#df.hist(figsize=(20,20))
#plt.show()

#Setting independant and target variables
X = df.drop(['Star color', 'Spectral Class'],axis=1)
y = df['Star type']
#Splitting Data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42,shuffle=True)

print(X.head())
print(y.head())

xgbc = XGBClassifier(max_depth=3,
                     subsample = 0.8,
                     n_estimators=200,
                     learning_rate=0.8,
                     min_child_weight=1,
                     reg_alpha=0,
                     reg_lambda=1
                     )
print(xgbc)

xgbc.fit(X_train, y_train)
y_predict = xgbc.predict(X_test)
y_train_predict = xgbc.predict(X_train)

# - cross validataion
scores = cross_val_score(xgbc, X_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbc, X_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

ypred = xgbc.predict(X_test)
cm = confusion_matrix(y_test,ypred)
print(cm)

print('train accuracy', accuracy_score(y_train, y_train_predict))
print('test accuracy', accuracy_score(y_test, ypred))



#Setting and fitting model
#model = XGBClassifier()
#model.fit(X, y)

#Feature importances and visualising it
#print(model.feature_importances_)

#feat_importances = pd.Series(model.feature_importances_, index = X.columns)
#feat_importances.nlargest(5).plot(kind = 'barh')
#plt.show()


#xgboostcl = XGBClassifier()
#xgboostcl.fit(X_train, y_train)

#Predictions
#preds = xgboostcl.predict(X_test)
#print("Predictions:\n", preds)
#print("\nTest Values:\n", y_test.values)
#print("\nAccuracy:", accuracy_score(preds,y_test.values))

#Pickling and dumping
#file = open('xgbcl_model.pkl', 'wb')
#pickle.dump(xgboostcl, file)

#checking git

