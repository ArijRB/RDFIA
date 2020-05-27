import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


data = np.load("data/15_scenes_Xy.npz", "rb")
X_train_all, X_test, y_train_all, y_test = train_test_split(data["X"], data["y"], test_size=.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=.125)


list_c = np.logspace(-6., 0., num=7)
score_test = []
score_train = []
best_score = -np.inf
best_c = None

for c in list_c:
    clf = LinearSVC(C=c, max_iter=int(1e2))
    clf.fit(X_train, y_train)
    score_train.append(clf.score(X_train, y_train))
    score = clf.score(X_val, y_val)
    print("c:", c, "score:", score)
    score_test.append(score)
    
    if score > best_score:
        best_score=score
        best_c=c

print("best c:", best_c)
list_c = np.log10(list_c)
plt.plot(list_c, score_train, label="train")
plt.plot(list_c, score_test, label="test")
plt.legend()
plt.plot()
plt.show()

clf = LinearSVC(C=best_c, max_iter=int(1e5))
clf.fit(X_train_all, y_train_all)
y_pred = clf.predict(X_test)
res = classification_report(y_test, y_pred)
print(res)
# test_score = clf.score(X_test, y_test)
# print("test_score", test_score)