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
    prev_pred = 'in'
    for xt in x_test_std:
        xt[160] = 1 if prev_pred == "in" else -1
        prev_pred = clf.predict(xt.reshape(-1, 1).T)
        y_test_pred.append(prev_pred)
    print(f"accuracy: {accuracy_score(y_test, y_test_pred)}")


def test_qualitative(modelfile, with_previous_state, with_bg):
    rt.detection(
        load(f'{macros.model_path}{modelfile}.joblib'),
        load(f'{macros.model_path}{modelfile}_scaler.joblib'),
        uses_previous_state=with_previous_state,
        with_bg = with_bg
    )
