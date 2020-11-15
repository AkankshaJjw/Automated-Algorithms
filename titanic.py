import csv as csv
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt

classifier = svm.LinearSVC()

def main():
    # read in train and test dataset
    train = pd.read_csv('train.csv', header=0)
    test = pd.read_csv('test.csv', header=0)
    train_data = data_clean(train)
    test_data = data_clean(test)
    # get labels outside dataset
    train_labels = train_data['Survived']
    train_data = train_data.drop(['Survived'], axis = 1)
    cross_validation_list = cross_validation(train_data, train_labels)
    false_negative = 0
    false_positive = 0
    for i in range(0, 5):
        FN, FP = train_classifier(cross_validation_list[i])
        false_negative += FN
        false_positive += FP
        
    false_negative = false_negative / 5
    false_positive = false_positive / 5
    print("\nfinal result")
    print(false_negative)
    print(false_positive)
    
   

    draw_fn_fp(0.1, 0.05, 0.2, 0.04, 0.3, 0.03, 0.4, 0.02)


def data_clean(data):
    '''
    get rid of Ticket, passenger id, name, Cabin, Embarked
    convert age and gender to float
    fill null age with -1
    fill fare with mean / -1
    '''
    data['Age'] = data['Age'].fillna(-1)
    data['Gender'] = data['Sex'].map({'female':0, 'male':1}).astype(int)
    data['Family'] = data['Parch'] + data['SibSp']
    # data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data['Fare'] = data['Fare'].fillna(-1)
    # data['Cabin'] = data['Cabin'].fillna(-1)
    # data['Cabin'] = data['Cabin'].fillna(-1)
    data = data.drop(['SibSp','Parch','PassengerId', 'Sex','Name','Cabin','Embarked','Ticket'],axis=1)
    return data

def draw_fn_fp(fp1, fn1, fp2, fn2, fp3, fn3, fp4, fn4):
    plt.plot([fp1,fp2,fp3,fp4],[fn1,fn2,fn3,fn4],'-ro',drawstyle='steps-post')
    plt.xlabel('false positive')
    plt.ylabel('false negative')
    plt.show()
    return

def cross_validation(data, train_labels):
    '''
    get 5 folders cross_validation
    return a list of 5 [data_train, data_test, label_train, label_test, train_index, test_index]
    list for each folder
    '''
    kf = KFold(n_splits=5)
    cross_validation_list = []
    for train_index, test_index in kf.split(data):
        data_train, data_test = data.iloc[train_index, :], data.iloc[test_index, :]
        label_train, label_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
        cross_validation_list.append([data_train, data_test, label_train, label_test, train_index, test_index])
    return cross_validation_list

def train_classifier(cross_list):
    '''
    train classifier and compare training results and testing results
    '''
    data_train_df, data_test_df, label_train_df, label_test_df, train_index, test_index = cross_list
    # get array values from pandas dataframe
    data_train = data_train_df.values
    data_test = data_test_df.values
    label_train = label_train_df.values
    label_test = label_test_df.values

    global classifier

    # classifier = svm.SVC(kernel = 'linear')
    # classifier.fit(data_train, label_train)

    # 0.2486703820416743, 0.1732139757578036
    #classifier = DecisionTreeClassifier(random_state=0, min_samples_split = 35, max_depth=40)
    #classifier.fit(data_train, label_train)

    # classifier = KNeighborsClassifier(n_neighbors=3, weights = 'uniform', algorithm = 'brute')
    # classifier.fit(data_train, label_train)

    # n_estimators = 4, 0.21544817556848228, 0.18389826532904807
    # n_estimators = 10, 0.24870124632212157, 0.17830743330743332
    # n_estimators = 15, 0.2562292790270062, 0.162512517004276
    # n_estimators = 4, balanced_subsample, 0.20073719652667016, 0.1920129258994717
    classifier = RandomForestClassifier(n_estimators = 100, min_samples_split = 10)
    classifier.fit(data_train, label_train)
    
    #classifier= QuadraticDiscriminantAnalysis(store_covariance=False)
    #classifier.fit(data_train, label_train)
    
    predicted_labels = classifier.predict(data_train)

    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(label_train, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(label_train, predicted_labels))
    print("F1 score: ", metrics.f1_score(label_train, predicted_labels, average='micro'))

    predicted_labels = classifier.predict(data_test)

    print("\nTesting results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(label_test, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(label_test, predicted_labels))
    print("F1 score: ", metrics.f1_score(label_test, predicted_labels, average='micro'))

    confusion_matrix = metrics.confusion_matrix(label_test, predicted_labels)
    '''
    Confusion matrix
                    Actual 0       Actual 1
    Predict 0       102 (TN)       17(FN)
    Predict 1       18 (FP)        42 (TP)
    '''
    FN = confusion_matrix[0, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
    FP = confusion_matrix[1, 0] / (confusion_matrix[1, 0] + confusion_matrix[0, 0])
    return FN, FP




if __name__ == '__main__':
    main()
