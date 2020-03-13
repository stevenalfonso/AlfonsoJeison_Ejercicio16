import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.tree

data = pd.read_csv('OJ.csv')
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0

data['Target'] = purchasebin
data = data.drop(['Purchase'],axis=1)
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')
#print(predictors)
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(np.array(data[predictors]), purchasebin, train_size = 0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

depth = np.arange(1,11)
F_train = np.zeros(10)
F_test = np.zeros(10)
std_train = np.zeros(10)
std_test = np.zeros(10)
ave_imp_mean = np.zeros((10,14))
ave_imp = np.zeros((100, 14))
for i in depth:
    clf = sklearn.tree.DecisionTreeClassifier(max_depth = i)
    f1_train = []
    f1_test = []
    for j in range(100):
        a = np.random.randint(0 , len(x_train), size = len(x_train))
        clf.fit(x_train[a,:], y_train[a])
        ave_imp[j,:] = clf.feature_importances_
        f1_train.append(sklearn.metrics.f1_score(y_train[a], clf.predict(x_train[a,:])))
        f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(x_test)))
    F_train[i-1] = np.mean(np.array(f1_train))
    F_test[i-1] = np.mean(np.array(f1_test))
    std_train[i-1] = np.std(np.array(f1_train))
    std_test[i-1] = np.std(np.array(f1_test))
    ave_imp_mean[i-1,:] = np.mean(ave_imp,0)

plt.errorbar(depth, F_train, std_train, fmt = 'o')
plt.errorbar(depth, F_test, std_test, fmt = 'o')
plt.xlabel('max depth')
plt.ylabel('Average F1-score')
plt.legend(['train 50%','test 50%'])
plt.savefig('F1_training_test.png')
plt.show()    

plt.figure()
Legends = []
for i in range(14):
    string = 'Col' + str(i)
    plt.plot(np.arange(1,11),ave_imp_mean[:,i])
    Legends.append(string)
plt.legend(Legends)
plt.ylabel('Average feature importance')
plt.xlabel('max depth')
plt.savefig('features.png')
plt.show()
