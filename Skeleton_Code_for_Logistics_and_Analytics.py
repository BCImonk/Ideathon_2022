##Libraries:
import os # Paths to file
import numpy as np # Linear Algebra
import pandas as pd # Data Processing
import warnings # Warning Filter

# Ploting Libraries:
import matplotlib.pyplot as plt 
import seaborn as sns

# Relevant ML Libraries:
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# ML Models:
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



##File Path:
# List all files under the input directory:
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Path for the training set:
tr_path = # TBD Later
# Path for the testing set:
te_path = # TBD Later



##Preprocessing and Data Analysis :
# Training Set:
# Read a csv file as a DataFrame:
tr_df = pd.read_csv(tr_path)
# Explore the first 5 rows:
tr_df.head()
# Testing Set:
# Read a csv file as a DataFrame:
te_df = pd.read_csv(te_path)
# Explore the first 5 rows:
te_df.head()
# Size of each data set:
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}")
# Column Information:
tr_df.info(verbose=True, null_counts=True)



##Data visalization:
'''We need to split our data to categorical and numerical data,
using the `.select_dtypes('dtype').columns.to_list()` combination.'''



##Region Demand Distribution:
# List of all the numeric columns:
num = tr_df.select_dtypes('number').columns.to_list()
# List of all the categoric columns:
cat = tr_df.select_dtypes('object').columns.to_list()
# Numeric df:
region_num =  tr_df[num]
# Categoric df:
region_cat = tr_df[cat]
print(tr_df[cat[-1]].value_counts())
total = float(len(tr_df[cat[-1]]))
plt.figure(figsize=(8,10))
sns.set(style="whitegrid")
ax = sns.countplot(tr_df[cat[-1]])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center") 
plt.show()



##Let's plot our data:
#Numeric:
for i in region_num:
    plt.hist(region_num[i])
    plt.title(i)
    plt.show()
# Categorical (split by Region Demand):
for i in cat[:-1]: 
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=i ,hue='Region_Status', data=tr_df ,palette='plasma')
    plt.xlabel(i, fontsize=14)



## Encoding data to numeric:
# Adding the new numeric values from the to_numeric variable to both datasets:
tr_df = tr_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
te_df = te_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable
# Checking our manipulated dataset for validation
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}\n")
print(tr_df.info(), "\n\n", te_df.info())
# Plotting the Correlation Matrix:
sns.heatmap(tr_df.corr() ,cmap='cubehelix_r')
# Correlation Table:
corr = tr_df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)



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



'''
Conclusion:
`Demand History` is a very important variable  because of its high correlation with `Region Demand` therefore showing high Dependancy for the latter.
'''



score = [DT_SC,RF_SC,XGB_SC,LR_SC]
Models = pd.DataFrame({
    'n_neighbors': ["Decision Tree","Random Forest","XGBoost", "Logistic Regression"],
    'Score': score})
Models.sort_values(by='Score', ascending=False)