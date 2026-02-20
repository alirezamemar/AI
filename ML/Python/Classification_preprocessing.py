import numpy as np 
import pandas as pd

df = pd.read_csv('./diabetes.csv')
df.head

df.shape
df['Outcomes'].values_count()

X = df.drop('Outcomes', axis=1)
y = df['Outcomes']

X
y

X=np.array(X)
y=np.array(y)

X
y

from sklearn.preprocessing import StandardScalar
scalar = StandardScalar()
X = scalar.fit_transform(X)

X


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape, y_train.shape
X_test.shape, y_test.shape
