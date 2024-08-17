import os

PATH_TO_DATASET = os.path.join("C:\\Users\\yozil\\Desktop\\My projects\\",
                               "10. End_to_End_Heart_Attack_Risk_Prediction",
                               "\\data\\raw data\\heart.csv")

TARGET = "output"

NUMERICAL_YEO_JOHNSON = ["trtbps"]

NUMERICAL_RECIPROCAL = ["chol"]

FEATURES = ['sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
            'exng', 'oldpeak', 'slp', 'caa', 'thall']