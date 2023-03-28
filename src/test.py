import src.real_time as rt
import src.models.dataset as dataset

from joblib import load
from sklearn.metrics import accuracy_score


def test_qualitative(modelfile):
    rt.detection(
        load(f'media/models/{modelfile}.joblib'),
        load(f'media/models/{modelfile}_scaler.joblib'))


def test_quantitative(testfiles, modelfile):
    clf = load(f'media/models/{modelfile}.joblib')
    scaler = load(f'media/models/{modelfile}_scaler.joblib')

    x_test, y_test = dataset.build_test(testfiles)
    x_test_std = scaler.transform(x_test)

    y_test_pred = []
    for xt in x_test_std:
        y_test_pred.append(clf.predict(xt.reshape(-1,1).T))

    print(f"accuracy: {accuracy_score(y_test, y_test_pred)}")


def test_quantitative_with_previous_state(testfiles, modelfile):
    clf = load(f'media/models/{modelfile}.joblib')
    scaler = load(f'media/models/{modelfile}_scaler.joblib')

    x_test, y_test = dataset.build_test(testfiles, True)
    x_test_std = scaler.transform(x_test)

    y_test_pred = []
    for xt in x_test_std:
        y_test_pred.append(clf.predict(xt.reshape(-1,1).T))
    print(f"accuracy: {accuracy_score(y_test, y_test_pred)}")


def test_qualitative_with_previous_state(modelfile):
    rt.detection(
        load(f'media/models/{modelfile}.joblib'),
        load(f'media/models/{modelfile}_scaler.joblib'),
        uses_previous_state=True
    )