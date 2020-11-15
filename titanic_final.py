import csv as csv
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

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
    FNFP = []
    classifiers = []
    classifiers.append([svm.SVC(kernel='poly', degree=5), 'svm5'])
    classifiers.append([svm.SVC(kernel='poly', degree=4), 'svm4'])
    classifiers.append([RandomForestClassifier(n_estimators = 100, min_samples_split = 10), 'rf'])
    classifiers.append([GradientBoostingClassifier(n_estimators=150, min_samples_split = 5), 'gb'])

    for classifier in classifiers:
        false_negative = 0
        false_positive = 0
        for i in range(5):
            FN, FP = train_classifier(cross_validation_list[i], classifier[0], classifier[1])
            false_negative += FN
            false_positive += FP
        FNFP.append(false_negative / 5)
        FNFP.append(false_positive / 5)
        print("\nfinal result")
        print(false_negative / 5)
        print(false_positive / 5)

    for i in range(len(FNFP)):
        if i % 2 == 0:
            print("___________________")
            print("false negative:" + str(FNFP[i]))
        else:
            print("false positive: " + str(FNFP[i]))

    predict_testing(classifiers, test, test_data)

    draw_fn_fp(FNFP)

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

def draw_fn_fp(fnfpArr):
    xArr = [fnfpArr[0],fnfpArr[2],fnfpArr[4],fnfpArr[6]]
    yArr = [fnfpArr[1],fnfpArr[3],fnfpArr[5],fnfpArr[7]]
    plt.plot(xArr, yArr, '-', drawstyle='steps-post')
    plt.xlabel('false negative')
    plt.ylabel('false positive')
    for i in range(4):
        plt.text(xArr[i] * 1.005, yArr[i] * 1.005, i)
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

def train_classifier(cross_list, classifier, model):
    '''
    train classifier and compare training results and testing results
    '''
    data_train_df, data_test_df, label_train_df, label_test_df, train_index, test_index = cross_list
    # get array values from pandas dataframe
    data_train = data_train_df.values
    data_test = data_test_df.values
    label_train = label_train_df.values
    label_test = label_test_df.values
    if model[:3] == 'svm':
        data_train = preprocessing.scale(data_train)
        data_test = preprocessing.scale(data_test)

    # 0.2486703820416743, 0.1732139757578036
    # classifier = DecisionTreeClassifier(random_state=0, min_samples_split = 3)
    # classifier.fit(data_train, label_train)

    classifier.fit(data_train, label_train)

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




def predict_testing(classifiers, test, test_data):
    for classifier in classifiers:
        if classifier[1][:3] == 'svm':
            print("yes")
            test_data = preprocessing.scale(test_data)
        predictions = pd.DataFrame(classifier[0].predict(test_data))
        predictions.columns = ['Survived']
        name = classifier[1] + ".csv"

        passId = test.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Gender', 'Family'],axis=1)
        formatted = pd.concat([passId, predictions], axis=1)


        formatted.to_csv(path_or_buf=name)




if __name__ == '__main__':
    main()