import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report , precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_wine
from sklearn.preprocessing import label_binarize
from scipy import interp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

start_date_and_time = datetime.now()
print(start_date_and_time)

#Load Dataset
dataset = pd.read_csv('scenario01.binetflow')
print(dataset.sample(5))
dataset = dataset[dataset['Label'].str.contains('Botnet')] #Getting the botnet data
print('Rows:',dataset.shape[0], 'Columns:', dataset.shape[1])
print(dataset.sample(5))

#Visualizing the Missing value
plt.figure(figsize=(16,5))
sns.set_style('whitegrid')
plt.title('% of Missing Values')
sns.barplot(x=dataset.isnull().mean().index, y=dataset.isnull().mean().values)
plt.xlabel('Columns')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.show()

# Removed all the columns with missing value > 30%
dataset = dataset.loc[:, dataset.isnull().mean() < 0.3] 

#Feature Engineering
'''
dataset = dataset.astype({"Proto":'category',"Sport":'category',"Dport":'category',"State":'category','StartTime':'datetime64[s]','StartTime':'datetime64[s]'}) # Changing the datatype of the columns
dataset['duration'] = abs(dataset['LastTime'].dt.second - dataset['StartTime'].dt.second) # getting duration from the columns 'LastTime' and 'StartTime'
dataset.drop(columns=['SrcAddr','DstAddr','LastTime','StartTime'],inplace=True) #Dropping the column SrcAddr and DstAddr since they contain unique ip addres
'''
#Data Visualization

#Analyzing categorical valriable
def barchart(columns):
    plt.figure(figsize=(10,5))
    plt.title(f'{columns}')
    sns.countplot(x=dataset[f'{columns}'].value_counts().values)
    plt.xlabel(f'{columns}')
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.show()
    
categorical_columns = dataset.select_dtypes(exclude=['int64', 'float64']).columns.values    
 
for column in categorical_columns:
    if column != 'Label':
        barchart(column)
        #pass
       
#Model Building
dataset = pd.get_dummies(dataset,columns=categorical_columns[:-1],drop_first=True)
X = dataset.loc[:, dataset.columns != 'Label']
y = dataset.loc[:, dataset.columns == 'Label']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=45)


# Decision Tree 
descision_tree_model = DecisionTreeClassifier()
descision_tree_model.fit(Xtrain,ytrain)
prediction_dt = descision_tree_model.predict(Xtest)
print('Decision Tree Accuracy Score:',round(accuracy_score(ytest,prediction_dt)*100),'%')
print('Decision Tree Precision Score:',round(precision_score(ytest,prediction_dt,average='micro')*100),'%')
print('Decision Tree Recall Score:',round(recall_score(ytest,prediction_dt,average='micro')*100),'%')
print('Decision Tree F1 Score:',round(f1_score(ytest,prediction_dt,average='micro')*100),'%')

'''
# SVM Classifier
support_vector_model = SVC()
support_vector_model.fit(Xtrain,ytrain)
prediction_svm = support_vector_model.predict(Xtest)
print('Support Vector Machine Accuracy Score:',round(accuracy_score(ytest,prediction_svm)*100),'%')
print('Support Vector Machine Precision Score:',round(precision_score(ytest,prediction_svm,average='micro')*100),'%')
print('Support Vector Machine Recall Score:',round(recall_score(ytest,prediction_svm,average='micro')*100),'%')
print('Support Vector Machine F1 Score:',round(f1_score(ytest,prediction_svm,average='micro')*100),'%')
'''

# Naive Bayes
multinomial_naive_bayes = GaussianNB()
multinomial_naive_bayes.fit(Xtrain,ytrain)
prediction_naive = multinomial_naive_bayes.predict(Xtest)
print('Naive Bayes Accuracy Score:',round(accuracy_score(ytest,prediction_naive)*100),'%')
print('Naive Bayes Precision Score:',round(precision_score(ytest,prediction_naive,average='micro')*100),'%')
print('Naive Bayes Recall Score:',round(recall_score(ytest,prediction_naive,average='micro')*100),'%')
print('Naive Bayes F1 Score:',round(f1_score(ytest,prediction_naive,average='micro')*100),'%')

# Logistic Regression
Logistic_model = LogisticRegression(C=1000)
Logistic_model.fit(Xtrain,ytrain)
prediction_Logistic = Logistic_model.predict(Xtest)
print('Logistic Regression Accuracy Score:',round(accuracy_score(ytest,prediction_Logistic)*100),'%')
print('Logistic Regression Precision Score:',round(precision_score(ytest,prediction_Logistic,average='micro')*100),'%')
print('Logistic Regression Recall Score:',round(recall_score(ytest,prediction_Logistic,average='micro')*100),'%')
print('Logistic Regression F1 Score:',round(f1_score(ytest,prediction_Logistic,average='micro')*100),'%')

# Random Forest
Random_forest_model = RandomForestClassifier(class_weight='balanced')
Random_forest_model.fit(Xtrain,ytrain)
prediction_Random_forest_model = Random_forest_model.predict(Xtest)
print('Random Forest Accuracy Score:',round(accuracy_score(ytest,prediction_Random_forest_model)*100),'%')
print('Random Forest Precision Score:',round(precision_score(ytest,prediction_Random_forest_model,average='micro')*100),'%')
print('Random Forest Recall Score:',round(recall_score(ytest,prediction_Random_forest_model,average='micro')*100),'%')
print('Random Forest F1 Score:',round(f1_score(ytest,prediction_Random_forest_model,average='micro')*100),'%')

end_date_and_time = datetime.now()
print(end_date_and_time)


'''
#ROC Curve and AUC
#X, y = load_wine(return_X_y=True)
#y = y == 2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
plt.show()

rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)

dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
ax = plt.gca()
dtc_disp = RocCurveDisplay.from_estimator(dtc, X_test, y_test, ax=ax, alpha=0.8)

nbc = GaussianNB()
nbc.fit(X_train, y_train)
ax = plt.gca()
nbc_disp = RocCurveDisplay.from_estimator(nbc, X_test, y_test, ax=ax, alpha=0.8)

lr = LogisticRegression()
lr.fit(X_train, y_train)
ax = plt.gca()
lr_disp = RocCurveDisplay.from_estimator(lr, X_test, y_test, ax=ax, alpha=0.8)

svc_disp.plot(ax=ax, alpha=0.8)

plt.show()

'''