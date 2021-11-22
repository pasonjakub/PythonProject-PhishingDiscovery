import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('dataset.csv', header=0)    # Data frame
df.drop('index', axis=1, inplace=True)
df.head()

# print(df.dtypes)


df['Result'].unique()

df_result = df[df['Result']==1]       # Legitimate
df_no_result = df[df['Result']==-1]   # Phishing

X = df.drop('Result', axis=1).copy()
# print(X.head())

y = df['Result'].copy()
# print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)


### Logistic Regression ###

from sklearn.linear_model import LogisticRegression

### Finding best operating parameters ###
# param_grid_lr = [{'penalty' : ['l1', 'l2'],'C' : np.logspace(-4, 4, 20),'solver' : ['liblinear']}]
# optimal_param_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, refit=True, verbose=3)
# optimal_param_lr.fit(X_train_scaled, y_train)
# optimal_param_lr.best_params_

# Logistic Regression model training
model_lr = LogisticRegression( penalty='l1', C = 0.08858667904100823, random_state=0, solver='liblinear')
model_lr.fit(X_train_scaled, y_train)
p_pred = model_lr.predict_proba(X_test_scaled)
y_pred = model_lr.predict(X_test_scaled)
print(model_lr.score(X_test_scaled,y_test))


### Decision Trees ###

from sklearn.tree import DecisionTreeClassifier            # for building classification tree
from sklearn.tree import plot_tree                         # for drawing classification tree
from sklearn.model_selection import train_test_split       # for splitting data into training and testing
from sklearn.model_selection import cross_val_score        # for cross validation

# Non-Optimized Decission Tree
model_dt = DecisionTreeClassifier(random_state = 42)
model_dt = model_dt.fit(X_train_scaled, y_train)

plt.figure(figsize = (20, 10))
plot_tree(model_dt, filled = True, rounded = True, class_names = ["Non-Phishing", "Phishing"], feature_names = X.columns);


### Support Vector Machines ###

from sklearn.svm import SVC

### Finding best operating parameters ###

# param_grid = [{'C':[0.1, 1, 10, 100, 1000], 'gamma':['scale', 1, 0.1, 0.01, 0.001, 0.0001], 'kernel':['rbf']}]
# optimal_param = GridSearchCV(SVC(), param_grid, cv=5, refit=True, verbose=3)
# optimal_param.fit(X_train_scaled, y_train)
# optimal_param.best_params_

model_svm = SVC(random_state=42, C=100, gamma=0.1, kernel = 'rbf', probability=True)
model_svm.fit(X_train_scaled, y_train)

### Comparison ###

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

pred_svm = model_svm.predict(X_test_scaled)

CM_svm = confusion_matrix(y_test, pred_svm)

svmAccuracy = (CM_svm[0][0] + CM_svm[1][1]) / (sum(map(sum, CM_svm)))   # (TrueNeg + TruePos) / (TrueNeg + FalseNeg + TruePos + FalsePos)
print(f"SVM Accuracy\t= {svmAccuracy}")
svmSensitivity = CM_svm[1][1] / (CM_svm[0][1] + CM_svm[1][1])           # TruePos / (TruePos + FalseNeg)
print(f"SVM Sensitivity\t= {svmSensitivity}")
svmSpecificity = CM_svm[0][0] / (CM_svm[0][0] + CM_svm[1][0])           # TrueNeg / (TrueNeg + FalsePos)
print(f"SVM Specificity\t= {svmSpecificity}")
svmPrecision = CM_svm[1][1] / (CM_svm[1][1] + CM_svm[1][0])             # TruePos / (TruePos + FalsePos)
print(f"SVM Precision\t= {svmPrecision}")
svmRecall = CM_svm[1][1] / (CM_svm[1][1] + CM_svm[0][1])                # TruePos / (TruePos + FalseNeg)
print(f"SVM Recall\t\t= {svmRecall}")

fig, ax = plt.subplots(figsize=(7.5, 7.5))
plot_confusion_matrix(model_svm, X_test_scaled, y_test, values_format='d', display_labels=["Not phishing","Phishing"], ax = ax)
plt.show()