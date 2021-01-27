#import modelling as mod
import dataPrep as dp
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def tunning_decision_tree(max_depth, X_train, X_test, y_train, y_test):
    best = {}
    criterion = ["gini","entropy"]
    splitter = ["best", "random"]
    for c in criterion:
        for s in splitter:
            for m in range(max_depth):
                dt = DecisionTreeClassifier(criterion=c, max_depth=m, splitter=s,random_state=0)
                dt.fit(X_train,y_train)
                score = dt.score(X_test,y_test)
                best[score] = DecisionTreeParams(c,m,s)
    key = max(best.keys())
    return best[key]

class DecisionTreeParams:
    def __init__(self, criterion, max_dept, splitter ):
        self.criterion = criterion
        self.max_depth = max_dept
        self.splitter = splitter

def tunning_random_forest():
    pass

def tunning_Ada_():
    pass

#print(tunning_decision_tree(7,ir_X_train,ir_X_test,ir_X_train,ir_y_test))
