import json
import os
from sklearn.externals import joblib
import pandas as pd
from azureml.core.model import Model
import pickle

def init():
    global model
    model_path = os.path.join(os.getnev('AZUREML_MODEL_DIR'), 'model.joblib')
    # Check if model is in location
    print("Found model: ", os.path.isfile(model_path))
    model = joblib.load(model_path)

def run(data):
    try:
        # Read in json data
        test_data = json.loads(data)['data']
        # Convert data to a Pandas DataFrame
        data_df = pd.DataFrame(test_data)
        # Make a prediction using loaded model
        result = model.predict(data_df)
        # Return result as json
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        error = str(e)
        return error