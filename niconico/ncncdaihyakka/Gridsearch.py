#Gridsearch.py
def PredictAndAnalyze3(data,target,clf_cv = svm.SVC(kernel='linear', probability=True),checkauc = False,ifprint = False,balancing = True):
    aucs = []
    y_trueall = []
    y_pridictall = []
    length = min([len(target[target == 0]),len(target[target == 1]),len(target[target == 2])])
    data = np.r_[data[target == 0][0:length],data[target == 1][0:length],data[target == 2][0:length]]
    target = np.r_[target[target == 0][0:length],target[target == 1][0:length],target[target == 2][0:length]]
    """
    length = min([len(target[target == 0]),len(target[target == 1])])
    data = np.r_[data[target == 0][0:length],data[target == 1][0:length]]
    target = np.r_[target[target == 0][0:length],target[target == 1][0:length]]
    """
    kf = KFold(len(target), n_folds=10, shuffle=True)
    #vmats = np.array([])
    for train, val in kf:
        X_train, y_train = np.array(data)[train], np.array(target)[train]
        X_test, y_test = np.array(data)[val], np.array(target)[val]
        clf_cv.fit(X_train, y_train)
        y_pred = clf_cv.predict(X_test)
        #vmat = clf_cv.coef_[0]
        """
        if vmats.shape[0] == 0:
            vmats = vmat
        else:
            vmats = np.c_[vmats,vmat]
        """
        y_true = y_test
        y_trueall = y_trueall + list(y_true)
        y_pridictall = y_pridictall  + list(y_pred)
        if ifprint == True:
            print(classification_report(y_true, y_pred))
        if checkauc == True:
            y_pred_cv = clf_cv.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_cv)
            aucs.append(auc)
    if checkauc == True:
        print np.mean(aucs), np.std(aucs)
    print(classification_report(y_trueall, y_pridictall))
    return y_trueall, y_pridictall#,vmats

from sklearn import grid_search
parameters = {
	'C': [2**(-9),2**(-5),2**(-1),2**(3),2**(7),2**(11),2**(15)],
	'gamma' : [2**(-11),2**(-7),2**(-3),2**(1),2**(3),2**(5),2**(9)]
}
length = min([len(target2[target2 == 0]),len(target2[target2 == 1]),len(target2[target2 == 2])])
data3 = np.r_[data2[target2 == 0][0:length],data2[target2 == 1][0:length],data2[target2 == 2][0:length]]
target3 = np.r_[target2[target2 == 0][0:length],target2[target2 == 1][0:length],target2[target2 == 2][0:length]]
clf = grid_search.GridSearchCV(svm.SVC(),parameters)
clf.fit(data3,target3)
print (clf.best_estimator_)
k = PredictAndAnalyze3(data = data2,target = target2,clf_cv = clf_cv)