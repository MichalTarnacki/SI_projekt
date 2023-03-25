import src.real_time as rt
import src.models.dataset as dataset

from joblib import load


def test_qualitative(modelfile):
    rt.detection(
        load(f'media/models/{modelfile}.joblib'),
        load(f'media/models/{modelfile}_scaler.joblib'))


def test_quantitative(testfiles, modelfile):
    clf = load(f'media/models/{modelfile}.joblib')
    scaler = load(f'media/models/{modelfile}_scaler.joblib')

    x_test, y_test = dataset.build_test(testfiles)
