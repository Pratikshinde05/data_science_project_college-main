# utils.py
import pandas as pd
import os
import pickle

def json_to_df(json_data):
    """
    Converts JSON data into a Pandas DataFrame.
    """
    try:
        df = pd.DataFrame(json_data)
        return df
    except Exception as e:
        raise ValueError(f"Error converting JSON to DataFrame: {str(e)}")
path=os.path.dirname("artifacts/json_to_df.pkl")

os.makedirs(path,exist_ok=True)

with open("artifacts/json_to_df.pkl", "wb") as f:
    pickle.dump(json_to_df, f)




