import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score, ConfusionMatrixDisplay

#load the data
data = pd.read_csv("C:/Users/acer/OneDrive - Universiti Malaya/Desktop/Phishing.csv")
ftnames = ["SFH","popUpWindow","SSLfinal_State","Request_URL","URL_of_Anchor","web_traffic","URL_Length","age_of_domain","having_IP_Address","Result"]
classnames = ["-1","0","1"]
print(data.head())

#EDA
#Examine the data
a=len(data[data.Result==0])
b=len(data[data.Result==-1])
c=len(data[data.Result==1])
print("Count of Suspicious Websites = ", a)
print("Count of Phishy Websites = ", b)
print("Count of Legitimate Websites = ", c)

#barplot
fig = plt.figure(figsize = (15,10))
for i in range(1,11):
    axi = fig.add_subplot(2,5,i)
    sns.countplot(data = data, x = ftnames[i-1], ax = axi, palette = 'pastel')
    axi.set_title(ftnames[i-1])
    axi.set_xlabel(" ")
plt.tight_layout()
plt.show()

#heatmap
sns.heatmap(data.corr(), annot = True)
plt.tight_layout()
plt.show()

#decision tree
#split data set in features and target variable
x = data.drop('Result',axis=1).values 
y = data['Result'].values

# Split dataset into training set and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=6) #Random 20% of full dataset as testing set
print("Training set has {} samples.".format(x_train.shape[0]))
print("Testing set has {} samples.".format(x_test.shape[0]))

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=6)

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Prune Decision Tree
dt_param_dist = {'max_leaf_nodes': range(2,100),
                 'max_depth': range(1,20),
                 'min_samples_leaf': list((30,100,300,int(0.05*(a+b+c))))}
dt_rand_search = RandomizedSearchCV(clf, param_distributions = dt_param_dist, n_iter=5, cv=5, random_state=6)
dt_rand_search.fit(x_train, y_train)
best_dt = dt_rand_search.best_estimator_
print('Best decision tree hyperparameters:',  dt_rand_search.best_params_)

#Visualizing the best decision tree
best_dt.fit(x_train,y_train)
fig = plt.figure(figsize=(12,30))
a = tree.plot_tree(best_dt, feature_names = ftnames, class_names = classnames, filled = True)
plt.show()

#Model Accuracy after Pruning
y_pred_best = best_dt.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred_best))

#random forest
#built random forest classifier with 50 decision trees
rf_model = RandomForestClassifier(n_estimators = 50, random_state=6)

#fit the model into training set
rf_model.fit(x_train, y_train)

#predict the response for the test set
y_pred = rf_model.predict(x_test)

#the model accuracy, precision & recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

#tunning the hyperparameters
rf_param_dist = {'n_estimators': range(50,2000),
                 'max_depth': range(1,30)}
rf_rand_search = RandomizedSearchCV(rf_model,param_distributions = rf_param_dist, n_iter=5, cv=5,random_state=6)
rf_rand_search.fit(x_train, y_train)
best_rf = rf_rand_search.best_estimator_
print('Best random forest hyperparameters:',rf_rand_search.best_params_)

#fit the best random forest model
best_rf.fit(x_train, y_train)

#predict & plot the confusion matrix
y_pred_best = best_rf.predict(x_test) 
cm=confusion_matrix(y_test,y_pred_best)
ConfusionMatrixDisplay(confusion_matrix=cm).plot( )
plt.show( )

#the model accuracy, precision and recall for the best model
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best, average="weighted")
recall = recall_score(y_test, y_pred_best, average="weighted")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

#importance of feature
best_rf.feature_importances_

#bar plot of feature importance
rounded_best_feature = [np.round(x,3) for x in best_rf.feature_importances_]
features = ftnames[:]
features.pop()
plt.barh(features, best_rf.feature_importances_)
for index, value in enumerate(rounded_best_feature):
    plt.text(value, index,
             str(value))
plt.show( )

