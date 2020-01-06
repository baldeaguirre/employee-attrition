import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

file = "TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx"

dataset = pd.read_excel(file, sheet_name="Existing employees")
dataset.rename(columns = {'dept':'department'}, inplace=True)
print(dataset.head(20))
print(dataset.dtypes, '\n')
print(dataset.isnull().any(), '\n')
print("Dataset shape:", dataset.shape, '\n')

# Departments
print(dataset['department'].unique(), '\n')
dataset['department'] = np.where(dataset['department'] == 'support', 'technical', dataset['department'])
dataset['department'] = np.where(dataset['department'] == 'IT', 'technical', dataset['department'])
print(dataset['department'].unique(), '\n')

print(dataset['left'].value_counts(), '\n')

# Checking the number of employees by 'department'
print(dataset['department'].value_counts(), '\n')

# Checking the number of employees by 'salary'
print(dataset['salary'].value_counts(), '\n')

# Mean of 'satisfaction_level', 'average_monthly_hours', 'Work_accident', 'promotion_last_5years'
print(dataset[['satisfaction_level', 'average_montly_hours', 'Work_accident', 'left', 'promotion_last_5years']].groupby('left').mean())

# Turnover Frequency for Department
pd.crosstab(dataset.department, dataset.left).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
plt.savefig('department_bar_chart')
plt.show()

# Stacked Bar Chart of Salary Level vs Turnover
table = pd.crosstab(dataset.salary, dataset.left)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.savefig('salary_bar_chart')
plt.show()

# Histogram of numeric variables
dataset['Emp ID'] = dataset['Emp ID'].astype('str')
num_bins = 10
dataset.hist(bins=num_bins, figsize=(20, 15))
plt.savefig("dataset_histogram_plots")
plt.show()

# Splitting categorical variables by columns ('department', 'salary')
cat_vars = ['department', 'salary']
for var in cat_vars:
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(dataset[var], prefix=var)
    dataset1 = dataset.join(cat_list)
    dataset = dataset1

# Removing categorical variables / columns
dataset.drop(dataset.columns[[0, 9, 10]], axis=1, inplace=True)
print(dataset.columns.values, '\n')

dataset_vars = dataset.columns.values.tolist()
y = ['left'] # Outcome variable
X = [i for i in dataset_vars if i not in y] # predictor variables

# Feature Selection (finding which variables are significant to predict employee attrition)
model = LogisticRegression()

rfe = RFE(model, 10)
rfe = rfe.fit(dataset[X], dataset[y])
#print(rfe.support_, '\n')
#print(rfe.ranking_, '\n')

sign_variables = np.where(rfe.support_ == True)
cols = np.array(X)[sign_variables]
print(cols, '\n')

X = dataset[cols]
y = dataset['left']

# Logistic Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))), '\n')

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, rf.predict(X_test))), '\n')

# Support Vector Machine
svc = SVC()
svc.fit(X_train, y_train)
print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_test, svc.predict(X_test))), '\n')

# Cross Validation
kfold = KFold(n_splits=10, random_state=7)
#kfold = KFold(n_splits=10)
modelCV = RandomForestClassifier()
scoring = 'accuracy'
results = cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()), '\n')

# Precision and recall / Random Forest
print(classification_report(y_test, rf.predict(X_test)))

y_pred = rf.predict(X_test)
forest_cm = confusion_matrix(y_pred, y_test, [1, 0])
sns.heatmap(forest_cm, annot=True, fmt='.2f', xticklabels=["Left", "Stayed"], yticklabels=["Left", "Stayed"])
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.savefig('random_forest')
plt.show()

# Logistic Regression
print(classification_report(y_test, logreg.predict(X_test)))

logreg_y_pred = logreg.predict(X_test)
logreg_cm = confusion_matrix(logreg_y_pred, y_test, [1, 0])
sns.heatmap(logreg_cm, annot=True, fmt='.2f', xticklabels=["Left", "Stayed"], yticklabels=["Left", "Stayed"])
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')
plt.savefig('logistic_regression')
plt.show()

#Support Vector Machine
print(classification_report(y_test, svc.predict(X_test)))

svc_y_pred = svc.predict(X_test)
svc_cm = confusion_matrix(svc_y_pred, y_test, [1, 0])
sns.heatmap(svc_cm, annot=True, fmt='.2f', xticklabels=["Left", "Stayed"], yticklabels=["Left", "Stayed"])
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Support Vector Machine')
plt.savefig('support_vector_machine')
plt.show()

# The ROC Curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])

rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()

# Feature Importance for Random Forest Model
feature_labels = np.array(['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years',
                           'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'])
importance = rf.feature_importances_
feature_indexes_by_importance = importance.argsort()

for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] * 100.0)))