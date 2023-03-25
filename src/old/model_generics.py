from joblib import dump, load


def save_model(model, filename):
    dump(model, filename)

def load_model(filename):
    return load(filename)
