import macros
import src.real_time as rt
import src.models.dataset as dataset

from joblib import load
from sklearn.metrics import accuracy_score


def test_quantitative(testfiles, modelfile, with_previous_state):
    clf = load(f'{macros.model_path}{modelfile}.joblib')
    scaler = load(f'{macros.model_path}{modelfile}_scaler.joblib')

    x_test, y_test = dataset.build(testfiles, macros.test_path, with_previous_state)
    x_test_std = scaler.transform(x_test)

    y_test_pred = []
    for xt in x_test_std:
        y_test_pred.append(clf.predict(xt.reshape(-1, 1).T))
    print(f"accuracy: {accuracy_score(y_test, y_test_pred)}")


def test_qualitative(modelfile, with_previous_state):
    rt.detection(
        load(f'{macros.model_path}{modelfile}.joblib'),
        load(f'{macros.model_path}{modelfile}_scaler.joblib'),
        uses_previous_state=with_previous_state
    )
