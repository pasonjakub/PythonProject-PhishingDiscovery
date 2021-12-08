import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
#print(model_lr.score(X_test_scaled,y_test))


### Decision Trees ###

from sklearn.tree import DecisionTreeClassifier            # for building classification tree
from sklearn.tree import plot_tree                         # for drawing classification tree
from sklearn.model_selection import train_test_split       # for splitting data into training and testing
from sklearn.model_selection import cross_val_score        # for cross validation


# Non-Optimized Decission Tree
# model_dt = DecisionTreeClassifier(random_state = 42)
# model_dt = model_dt.fit(X_train_scaled, y_train)
#
# plt.figure(figsize = (20, 10))
# plot_tree(model_dt, filled = True, rounded = True, class_names = ["Non-Phishing", "Phishing"], feature_names = X.columns);
#
# #Searching for best parameters
#
# path = model_dt.cost_complexity_pruning_path(X_train_scaled, y_train)           #determine values for alpha
# ccp_alphas = path.ccp_alphas                                                    #extract different values for alpha
# ccp_alphas = ccp_alphas[:-1]                                                    #exclude the maximum value for alpha
#
# model_dt_a = []                                                                 #new array for storing the tree
#
# for ccp_alpha in ccp_alphas:
#   model_dt = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
#   model_dt.fit(X_train_scaled, y_train)
#   model_dt_a.append(model_dt)
#
# train_scores = [model_dt.score(X_train_scaled, y_train) for model_dt in model_dt_a]
# test_scores = [model_dt.score(X_test_scaled, y_test) for model_dt in model_dt_a]
#
# fig, alphax = plt.subplots(figsize = (10,5))
# alphax.set_xlabel("alpha")
# alphax.set_ylabel("accuracy")
# alphax.set_title("Accuracy vs alpha")
# alphax.plot(ccp_alphas, train_scores, marker = 'o', label = "train", drawstyle = "steps-post")
# alphax.plot(ccp_alphas, test_scores, marker = 'o', label = "test", drawstyle = "steps-post")
# alphax.legend()
# plt.show()
#
# model_dt = DecisionTreeClassifier(random_state = 42, ccp_alpha = 0.0005) #create the tree with alpha = 0.0005
# scores = cross_val_score(model_dt, X_train_scaled, y_train, cv = 5)
# data_frame = pd.DataFrame(data = {'Tree': range(5), 'accuracy': scores})
#
# data_frame.plot(x = 'Tree', y = 'accuracy', marker = 'o', linestyle = '--')
#
#
# alpha_loop_values = [] # an array to store alphas
#
# for ccp_alpha in ccp_alphas:
#   model_dt = DecisionTreeClassifier(random_state = 0, ccp_alpha=ccp_alpha)
#   scores = cross_val_score(model_dt, X_train_scaled, y_train, cv = 5)
#   alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
#
# alpha_results = pd.DataFrame(alpha_loop_values, columns = ['alpha', 'mean_accuracy', 'std'])
#
# alpha_results.plot(figsize = (10,5), x = 'alpha', y = 'mean_accuracy', yerr = 'std', marker = 'o', linestyle = '--')
#
# alpha_results[(alpha_results['alpha'] > 0.00004) &  (alpha_results['alpha'] < 0.00007)]

ideal_ccp_alpha = 0.000051


model_dt = DecisionTreeClassifier(random_state = 42, ccp_alpha=ideal_ccp_alpha)
model_dt = model_dt.fit(X_train_scaled, y_train)

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

# Logistic Regression
y_pred_lr = model_lr.predict(X_test_scaled)

lr_cm = confusion_matrix(y_test,y_pred_lr)
lr_true_neg = lr_cm[0][0]
lr_false_neg = lr_cm[0][1]
lr_true_pos = lr_cm[1][1]
lr_false_pos = lr_cm[1][0]

lr_accuracy = (lr_true_pos + lr_true_neg)/(lr_true_pos + lr_true_neg + lr_false_pos + lr_false_neg)
print('LR Accuracy\t\t=',lr_accuracy)
lr_sensitivity = lr_true_pos/(lr_true_pos + lr_false_neg)
print('LR Sensitivity\t=',lr_sensitivity)
lr_specificity = lr_true_neg/(lr_true_neg + lr_false_pos) #more important than sensitivity? (false alarm)
print('LR Specificity\t=',lr_specificity)
lr_precision = lr_true_pos/(lr_true_pos + lr_false_pos)
print('LR Precision\t=',lr_precision)
lr_recall = lr_true_pos/(lr_true_pos + lr_false_neg)
print('LR Recall\t\t=',lr_recall)

cm_lr = confusion_matrix(y_test, y_pred_lr)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.imshow(cm_lr)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Not phishing', 'Predicted Phishing'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Not Phishing', 'Actual Phishing'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_lr[i, j], ha='center', va='center', color='red')

#Support vector machines
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

_, ax = plt.subplots(figsize=(7.5, 7.5))
plot_confusion_matrix(model_svm, X_test_scaled, y_test, values_format='d', display_labels=["Not phishing","Phishing"], ax = ax)


#Decision Trees

_, ax = plt.subplots(figsize=(7.5, 7.5))
plot_confusion_matrix(model_dt, X_test_scaled, y_test, values_format='d', display_labels = ["Non-Phishing", "Phishing"], ax = ax)

plt.figure(figsize = (20, 10))
plot_tree(model_dt, filled = True, rounded = True, class_names = ["Non-Phishing", "Phishing"], feature_names = X.columns)

dt_y_pred = model_dt.predict(X_test_scaled)

dt_cm = confusion_matrix(y_test,dt_y_pred)
dt_true_neg = dt_cm[0][0]
dt_false_neg = dt_cm[0][1]
dt_true_pos = dt_cm[1][1]
dt_false_pos = dt_cm[1][0]

dt_accuracy = (dt_true_pos + dt_true_neg)/(dt_true_pos + dt_true_neg + dt_false_pos + dt_false_neg)
print('DT Accuracy\t\t=',dt_accuracy)
dt_sensitivity = dt_true_pos/(dt_true_pos + dt_false_neg)
print('DT Sensitivity\t=',dt_sensitivity)
dt_specificity = dt_true_neg/(dt_true_neg + dt_false_pos) # more important than sensitivity? (false alarm)
print('DT Specificity\t=',dt_specificity)
dt_precision = dt_true_pos/(dt_true_pos + dt_false_pos)
print('DT Precision\t=',dt_precision)
dt_recall = dt_true_pos/(dt_true_pos + dt_false_neg)
print('DT Recall\t\t=',dt_recall)

# Receiving operating characteristic
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score

r_probs = [0 for _ in range (len(y_test))]  # random
lr_probs = model_lr.predict_proba(X_test_scaled)[:,1]
svm_probs = model_svm.predict_proba(X_test_scaled)[:,1]
dt_probs = model_dt.predict_proba(X_test_scaled)[:,1]

r_auc = roc_auc_score(y_test, r_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
svm_auc = roc_auc_score(y_test, svm_probs)
dt_auc = roc_auc_score(y_test, dt_probs)

r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)    # false positive rate, true positive rate
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)

plt.figure(figsize=(10,5))
plt.plot(r_fpr, r_tpr, linestyle = '--', label = 'Random prediction: AUROC = %.3f' %r_auc)
plt.plot(lr_fpr, lr_tpr, linestyle='-', label='Logistic regression: AUROC = %0.3f' % lr_auc)
plt.plot(svm_fpr, svm_tpr, linestyle = '-', label = 'Support Vector Machines: AUROC = %.3f' %svm_auc)
plt.plot(dt_fpr, dt_tpr, linestyle='-', label='Decision Tree: AUROC = %0.3f' % dt_auc)

plt.title('Receiver operating characteristic')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()



svm_y_score = model_svm.decision_function(X_test_scaled)
svm_average_pr = average_precision_score(y_test, svm_y_score)
lr_y_score = model_lr.decision_function(X_test_scaled)
lr_average_pr = average_precision_score(y_test, lr_y_score)
dt_average_pr = precision_score(dt_y_pred, y_test, average='macro')

r_r, r_p, _ = precision_recall_curve(y_test, r_probs)
svm_r, svm_p, _ = precision_recall_curve(y_test, svm_probs)
lr_r, lr_p, _ = precision_recall_curve(y_test, lr_probs)
dt_r, dt_p, _ = precision_recall_curve(y_test, dt_probs)

plt.figure(figsize=(10,5))
plt.plot(svm_p, svm_r, linestyle = '-', label = 'SVM:  = %.3f' %svm_average_pr)
plt.plot(lr_p, lr_r, linestyle = '-', label = 'Logistic regression:  = %.3f' %lr_average_pr)
plt.plot(dt_p, dt_r, linestyle = '-', label = 'Decision Tree:  = %.3f' %dt_average_pr)

plt.title('Precision Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

plt.show()