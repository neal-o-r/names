import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import xgboost


def get_data():

        df = pd.read_csv('names.txt')

        # shuffle data        
        df = df.sample(frac=1.).reset_index(drop=True)
        
        df['Length'] = df.Name.apply(len)
        df['First'] = df.Name.apply(lambda x: x[0].lower())
        df['Last'] = df.Name.apply(lambda x: x[-1])

        le = preprocessing.LabelEncoder()
        df = df.apply(le.fit_transform)

        X = np.array(df[['Length', 'First', 'Last']])

        Y = np.array(df.Class)

        return df, X, Y


if __name__ == '__main__':

        df, X, Y = get_data()

        n = int(0.1 * len(Y))

        X_train = X[:9*n]
        X_test  = X[9*n:]
        Y_test  = Y[9*n:]
        Y_train = Y[:9*n]
        
        model = tree.DecisionTreeClassifier()

        model.fit(X_train, Y_train)

        acc = accuracy_score(Y_test, model.predict(X_test))

        print("Tree Accuracy on test set: %.2f%%" %(100*acc))
       

        forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

        forest.fit(X_train, Y_train)

        acc = accuracy_score(Y_test, forest.predict(X_test))
        print("Forest Accuracy on test set: %.2f%%" %(100*acc))

        xgb = xgboost.XGBClassifier()

        xgb.fit(X_train, Y_train)
        acc = accuracy_score(Y_test, xgb.predict(X_test))
        print("XGBoost Accuracy on test set: %.2f%%" %(100*acc))


