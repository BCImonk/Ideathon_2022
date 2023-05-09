'''
## First of all we will divide our dataset into two variables `X` as the features we 
defined earlier and `y` as the `Region_Demand` the target value we want to predict.

## Machine Learning Models we will use:

* **Decision Tree** 
* **Random Forest**
* **XGBoost**
* **Logistic Regression**

## The Process of Modeling the Data:
1. Importing the model
2. Fitting the model
3. Predicting Region Demand
4. Classification report by Region Demand
5. Overall accuracy
'''



y = tr_df['Region Demand']
X = tr_df.drop('Region Demand', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

##Decision Tree :
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
y_predict = DT.predict(X_test)
# Prediction Summary by species :
print(classification_report(y_test, y_predict))
# Accuracy Score :
DT_SC = accuracy_score(y_predict,y_test)
print(f"{round(DT_SC*100,2)}% Accurate")
# Csv results of the test for our model :
# (You can see each predition and true value side by side by the csv created in the output directory.)
Decision_Tree=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Decision_Tree.to_csv("Dection Tree.csv")     



##Random Forest :
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict = RF.predict(X_test)
#  Prediction Summary by species:
print(classification_report(y_test, y_predict))
# Accuracy score:
RF_SC = accuracy_score(y_predict,y_test)
print(f"{round(RF_SC*100,2)}% Accurate")
# Csv results of the test for our model:
# (You can see each predition and true value side by side by the csv created in the output directory.)
Random_Forest=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Random_Forest.to_csv("Random Forest.csv")     



##XGBoost:
XGB = XGBClassifier()
XGB.fit(X_train, y_train)
y_predict = XGB.predict(X_test)
# Prediction Summary by species:
print(classification_report(y_test, y_predict))
# Accuracy Score:
XGB_SC = accuracy_score(y_predict,y_test)
print(f"{round(XGB_SC*100,2)}% Accurate")
#Csv results of the test for our model:
#(You can see each predition and true value side by side by the csv created in the output directory.)
XGBoost=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
XGBoost.to_csv("XGBoost.csv")     



##Logistic Regression:
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_predict = LR.predict(X_test)
# Prediction Summary by species:
print(classification_report(y_test, y_predict))
# Accuracy Score:
LR_SC = accuracy_score(y_predict,y_test)
print('accuracy is',accuracy_score(y_predict,y_test))
Logistic_Regression=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Logistic_Regression.to_csv("Logistic Regression.csv")     