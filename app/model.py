import pickle

def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def load_scaler():
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler