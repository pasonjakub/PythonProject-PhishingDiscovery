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