import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

with open('des_sMCI.npy', 'rb') as file:
    s = np.load(file)

with open('des_pMCI.npy', 'rb') as file:
    p = np.load(file)

# cut to the same length
min_len = min(len(min(s, key=len)), len(min(p, key=len)))
X = []; y = []
for i in range(len(s)):
    X.append(s[i][:min_len])
    y.append(0)

for i in range(len(p)):
    X.append(p[i][:min_len])
    y.append(1)

X = np.array(X); y = np.array(y)

# preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# training
#without random forest
pca = PCA(n_components=50) # decomposition
X1 = pca.fit_transform(X)

# search
# best: C=1, kernel='linear'

cv = ShuffleSplit(n_splits=4, test_size=0.25)
search = GridSearchCV(svm.SVC(), cv=cv, param_grid=[
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']}])
search.fit(X1, y)
print('best accuracy:', search.best_score_)
print('best params:', search.best_params_)



cplt = np.arange(0.5, 50, 0.5)
score1 = []
for i, C in enumerate(np.arange(0.5, 50, 0.5)):
    clf_svm = svm.SVC(C=C, cache_size=200, kernel='linear')
    cv = ShuffleSplit(n_splits=4, test_size=0.25) # choose randomly from s and p
    score = cross_val_score(clf_svm, X1, y, cv=cv) # for small dataset
    score1.append(score.mean())
    print('C:', C, 'accuracy:', score.mean())


#with random forset
rf = RandomForestClassifier(n_estimators=1000, random_state=111)
clt = rf.fit(X, y)
fi = clt.feature_importances_
index = np.where(fi>=(0.5*np.mean(fi)))
X = np.squeeze(X[:,index], axis=(1,))


pca = PCA(n_components=50) # decomposition
X = pca.fit_transform(X)


score2 = []
for i, C in enumerate(np.arange(0.5, 50, 0.5)):
    clf_l2_LR = svm.SVC(C=C, cache_size=100, kernel='linear')
    cv = ShuffleSplit(n_splits=4, test_size=0.25) # choose randomly from s and p
    score = cross_val_score(clf_l2_LR, X, y, cv=cv) # for small dataset
    score2.append(np.mean(score))
    print('C:', C, 'accuracy:', score.mean())


plt.plot(cplt, score1, label='PCA')
plt.plot(cplt, score2, label='PCA+feature_importance')
plt.xlabel('Regularization Strength')
plt.ylabel('Accuracy')
plt.title(r'Accuracy using SVM')
plt.legend()
plt.show()
