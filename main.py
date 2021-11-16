import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#Logistic Regression Grid Search
# param_grid_lr = [{'penalty' : ['l1', 'l2'],'C' : np.logspace(-4, 4, 20),'solver' : ['liblinear']}]
# optimal_param_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, refit=True, verbose=3)
# optimal_param_lr.fit(X_train_scaled, y_train)
# optimal_param_lr.best_params_

#Logistic Regression model training
model = LogisticRegression( penalty='l1', C = 0.08858667904100823, random_state=0, solver='liblinear')
model.fit(X_train_scaled, y_train)
p_pred = model.predict_proba(X_test_scaled)
y_pred = model.predict(X_test_scaled)
print(model.score(X_test_scaled,y_test))
