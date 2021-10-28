import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
#----------------------------------------------------------------------------------------
# Read in Data
#----------------------------------------------------------------------------------------
cancer = datasets.load_breast_cancer() # 0 represents malignant, 1 represents benign
#print(cancer.feature_names)
#print(cancer.target_names)
#-----------------------------------------------------------------------------------------
# Creating the Model
#-----------------------------------------------------------------------------------------
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# Test based on x_train, x_test, y_train, y_test

classes = ['malignant' 'benign']
#------------------------------------------------------------------------------------------
# Doing the prediction
#------------------------------------------------------------------------------------------
clf = svm.SVC(kernel='linear', C=2) # SVC stands for support vector classification | C= soft margin
clf.fit(x_train, y_train)

y_prediction = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_prediction)
print(acc)