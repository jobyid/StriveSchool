from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import dataPrep as dp
import model_tuning as mt
from sklearn import tree
import matplotlib.pyplot as plt

ir_X_train, ir_X_test, ir_y_train, ir_y_test, fn, cn = dp.preped_iris()
z_X_train, z_X_test, z_y_train, z_y_test, zfn, zcn = dp.prepped_zoo()
c_X_train, c_X_test, c_y_train, c_y_test, cfn, ccn = dp.prepped_cars()
b_X_train, b_X_test, b_y_train, b_y_test, bfn, bcn = dp.prepped_bank_notes()

criterion = 'gini'
max_depth = 5
splitter = 'best'
f_n = ["Sepal Length","Sepal Width", "Petal Length", "Petal Width"]
c_n = ['setosa', 'versicolor', 'virginica']

def run_decision_tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=0,criterion=criterion, max_depth=max_depth, splitter=splitter)
    dt.fit(X_train,y_train)
    pred = dt.predict(X_test)
    score = dt.score(X_test, y_test)
    #path = dt.decision_path(X_test)
    params = dt.get_params()
    tree.plot_tree(dt,filled=True,feature_names=f_n, class_names=c_n, rounded=True)
    plt.show()
    return score, pred, params

n_estimators = 100
def run_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X_train,y_train)
    pred = rf.predict(X_test)
    score = rf.score(X_test,y_test)
    return score, pred

def run_ada_boost(X_train, X_test, y_train, y_test):
    ad = AdaBoostClassifier()
    ad.fit(X_train, y_train)
    pred = ad.predict(X_test)
    score = ad.score(X_test, y_test)
    return score, pred

def run_graident_boost(X_train, X_test, y_train, y_test):
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    score = gb.score(X_test, y_test)
    return score, pred

def print_results(dataset_name, X_train, X_test, y_train, y_test, fn, cn):
    global f_n
    global c_n
    f_n = fn
    c_n = cn
    print("--- ",dataset_name," ---")
    print("Decision Tree")
    print("Decision tree score on ", dataset_name,": ", run_decision_tree(X_train, X_test, y_train, y_test)[0])
    print("Random Forest")
    print("Random Forest score on ", dataset_name,": ",run_random_forest(X_train, X_test, y_train, y_test)[0])
    print("Ada Boost")
    print("Ada Boost score on ", dataset_name,": ",run_ada_boost(ir_X_train, ir_X_test, ir_y_train, ir_y_test)[0])
    print("Gradient Boost")
    print("Gradient Boost score on ", dataset_name,": ",run_graident_boost(X_train, X_test, y_train, y_test)[0], end="\n\n")

print_results("Iris Data", ir_X_train, ir_X_test, ir_y_train, ir_y_test, fn, cn)
print_results("Zoo Data", z_X_train, z_X_test, z_y_train, z_y_test, zfn, zcn)
print_results("Car data", c_X_train, c_X_test, c_y_train, c_y_test,cfn,ccn)
print_results("Banknote Data", b_X_train, b_X_test, b_y_train, b_y_test,bfn,bcn)

#run_decision_tree(ir_X_train,ir_X_test,ir_y_train,ir_y_test)
