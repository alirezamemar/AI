from sklearn.naive_byes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

y_predict_train = model.predict(X_train)
y_predict_test = model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
c = confusion_matrix(y_test, y_pred_test)
a_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
a_test = accuracy_score(y_true=t_test, y_pred=y_pred_test)
p = precision_score(y_test, y_pred_test)
r = recall_score(y_test, y_pred_test)
