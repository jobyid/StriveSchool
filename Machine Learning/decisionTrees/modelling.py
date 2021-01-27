from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import dataPrep as dp


def run_decision_tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train,y_train)
    pred = dt.predict(X_test)
    score = dt.score(X_test, y_test)
    return score, pred

print("Decision Tree")
ir_X_train, ir_X_test, ir_y_train, ir_y_test = dp.preped_iris()
print("Decision tree score on iris: ", run_decision_tree(ir_X_train, ir_X_test, ir_y_train, ir_y_test)[0])
z_X_train, z_X_test, z_y_train, z_y_test = dp.prepped_zoo()
print("Decision tree score on zoo: ", run_decision_tree(z_X_train, z_X_test, z_y_train, z_y_test)[0])
c_X_train, c_X_test, c_y_train, c_y_test = dp.prepped_cars()
print("Decision tree score on cars: ", run_decision_tree(c_X_train, c_X_test, c_y_train, c_y_test)[0])
b_X_train, b_X_test, b_y_train, b_y_test = dp.prepped_bank_notes()
print("Decision tree score on banknotes: ", run_decision_tree(b_X_train, b_X_test, b_y_train, b_y_test)[0], end="\n\n")

def run_random_forest(X_train, X_test, y_train, y_test,estimators):
    rf = RandomForestClassifier(n_estimators=estimators)
    rf.fit(X_train,y_train)
    score = rf.score(X_test,y_test)
    return score,0

n_estimators = 100
print("Random Forest")
ir_X_train, ir_X_test, ir_y_train, ir_y_test = dp.preped_iris()
print("Random Forest score on iris: ", run_random_forest(ir_X_train, ir_X_test, ir_y_train, ir_y_test,n_estimators)[0])
z_X_train, z_X_test, z_y_train, z_y_test = dp.prepped_zoo()
print("Random Forest score on zoo: ", run_random_forest(z_X_train, z_X_test, z_y_train, z_y_test,n_estimators)[0])
c_X_train, c_X_test, c_y_train, c_y_test = dp.prepped_cars()
print("Random Forest score on cars: ", run_random_forest(c_X_train, c_X_test, c_y_train, c_y_test,n_estimators)[0])
b_X_train, b_X_test, b_y_train, b_y_test = dp.prepped_bank_notes()
print("Random Forest score on banknotes: ", run_random_forest(b_X_train, b_X_test, b_y_train, b_y_test,n_estimators)[0])
