from enum import Enum
from sklearn import svm
from sklearn import tree
from sklearn import naive_bayes


class ModelType(Enum):
    SVC = 'SVC'
    GuassianNB = 'GuassianNB'
    DecisionTree = 'DecisionTree'


def get_classifier(name=ModelType.SVC, C=1.0, kernel='rbf'):
    if name == ModelType.SVC:
        return svm.SVC(C=C, kernel=kernel)
    elif name == ModelType.GuassianNB:
        return naive_bayes.GaussianNB()
    elif name == ModelType.DecisionTree:
        return tree.DecisionTreeClassifier()
    else:
        raise NotImplementedError


