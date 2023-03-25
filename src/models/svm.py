import src.models.dataset as dataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import dump


class SVM:
    def fit(self, x_train, y_train):
        pass

    def predict(self):
        pass


def library_svm_train(filenames, modelname):
    x_train, y_train = dataset.build_train(filenames)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    clf = SVC(kernel='linear', verbose=1)
    clf.fit(x_train_std, y_train)
    dump(clf, f'media/models/{modelname}.joblib')
    dump(scaler, f'media/models/{modelname}_scaler.joblib')

    return clf, scaler
