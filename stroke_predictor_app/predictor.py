import pickle
import pandas as pd
import numpy as np



class stroke_predictor():
    def __init__(self):
        pass
    def deserialize(self):
        with open('clf.pkl', 'rb') as handle:
            model = pickle.load(handle)
            return model
        
    def predict(self, age, hypertension, heart_disease, avg_glucose_level, bmi):
        model = self.deserialize()
        return model.predict(np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]]))
