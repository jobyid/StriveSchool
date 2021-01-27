from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def preped_iris():
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    fn = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    cn = ['setosa', 'versicolor', 'virginica']
    return X_train, X_test, y_train, y_test, fn, cn

def prepped_zoo():
    df = pd.read_csv('./data/zoo.data')
    columns = []
    with open("./data/zoo.names") as f:
        s = f.read().split("7. Attribute Information: (name of attribute and type of value domain)")
        s = s[1].split("8. Missing Attribute Values: None")
        s = s[0].replace('Boolean', '').replace('Numeric (set of values: {0,2,4,5,6,8})', '').replace('Numeric (integer values in range [1,7])','').replace(':      Unique for each instance','').replace('\t','')
        columns = s.strip().split("\n")
    df.columns = columns
    y = df.iloc[:,-1]
    X = df.iloc[:,1:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    fn = columns
    cn = ["1","2",'3','4','5','6','7']
    return X_train, X_test, y_train, y_test, fn, cn

def prepped_cars():
    df = pd.read_csv('./data/car.data')
    df.columns = ['buying','maint','doors','persons', 'lug_boot', 'safety','target']
    df = pd.get_dummies(df)
    y = df.iloc[:, -1]
    X = df.iloc[:, 1:-4]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    fn = ['buying_high', 'buying_low', 'buying_med', 'buying_vhigh', 'maint_high',
       'maint_low', 'maint_med', 'maint_vhigh', 'doors_2', 'doors_3',
       'doors_4', 'doors_5more', 'persons_2', 'persons_4', 'persons_more',
       'lug_boot_big', 'lug_boot_med', 'lug_boot_small', 'safety_high',
       'safety_low', 'safety_med']
    cn = ['unacc', 'acc', 'good', 'vgood']
    return X_train, X_test, y_train, y_test, fn,cn

def prepped_bank_notes():
    df = pd.read_csv('./data/data_banknote_authentication.txt')
    df.columns = ['vari','skew','curto','entro','target']
    y = df.iloc[:, -1]
    X = df.iloc[:, 1:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    fn = ['vari','skew','curto','entro','target']
    cn = ['fake', 'real']
    return X_train, X_test, y_train, y_test, fn, cn

