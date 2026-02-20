from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

c = confusion_matrix(y_test, y_pred_test)
a = accuracy_score(y_test, y_pred_test)
p = precision_score(y_test, y_pred_test)
r = recall_score(y_test, y_pred_test)
