import pickle

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_category(resume_text):
    vec = vectorizer.transform([resume_text])
    return model.predict(vec)[0]