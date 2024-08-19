# importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

class Pipeline:
    def __init__(self,
                 target,
                 num_reciprocal,
                 num_yeo_johnson,
                 features,
                 test_size=0.1,
                 random_state=42
                 ):
        
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.num_reciprocal =num_reciprocal
        self.num_yeo_johnson = num_yeo_johnson
        self.features = features
        
        # initialize the split datasets
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
        # models
        self.scaler = StandardScaler()
        self.model = LogisticRegression()
        
        # store yeo johnsont parameter for transfomation
        self.yeo_johnson_dict = {}
    
    # ============================================================#
    # Functions to serve Feature Engineering #
    def fit(self, data):
        # split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data.drop(self.target, axis =1),
                                                                                data[self.target],
                                                                                test_size = self.test_size,
                                                                                random_state=self.random_state)
        
        # apply a reciprocal transformation on the trtbps feature
        for var in self.num_reciprocal:
            self.x_train[var] = 1/self.x_train[var]
            self.x_test[var] = 1/self.x_test[var]
        
        # now apply a yeo johnoson transformation on the train set
        for var in self.num_yeo_johnson:
            self.x_train[var], param = stats.yeojohnson(self.x_train[var])
            self.yeo_johnson_dict[var] = param
            self.x_test[var] = stats.yeojohnson(self.x_test[var], lmbda=param)

        
        # now apply a standard scaler to the whole dataframe
        self.scaler.fit(self.x_train)
        
        self.x_train = pd.DataFrame(self.scaler.transform(self.x_train), columns = self.x_train.columns)
        self.x_test = pd.DataFrame(self.scaler.transform(self.x_test), columns = self.x_test.columns)
        
        # train model
        self.model.fit(self.x_train[self.features], self.y_train)
        
        print("Model Trained Successfully!!!")
        
        return self
    
    def transform(self, data):
        data = data.copy()

        # apply reciprocal transformation
        for var in self.num_reciprocal:
            data[var] = 1/data[var]
            
        # apply the yeo johnson transformation
        for var in self.num_yeo_johnson:
            data[var] = stats.yeojohnson(data[var],
                                         lmbda=self.yeo_johnson_dict[var])
            
        # apply the scaler
        data_transformed = pd.DataFrame(self.scaler.transform(data), columns = data.columns)
        
        return data_transformed
    
    def predict(self, data):
        # transform the new data
        data_transformed = self.transform(data)
        
        # make predictions
        predictions = self.model.predict(data_transformed[self.features])
        
        return predictions
    
    def evaluate(self):
        
        print()
        print("Training Evaluation Results")
        print()
         # make predictions on the test set
        tr_predictions = self.model.predict(self.x_train[self.features])
        
        # calculate evaluation metrics
        accuracy = accuracy_score(self.y_train, tr_predictions)
        precision = precision_score(self.y_train, tr_predictions, average='binary')
        recall = recall_score(self.y_train, tr_predictions, average='binary')
        f1 = f1_score(self.y_train, tr_predictions, average='binary')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print()
        print()
        print("Test Evaluation Results")
        print()
        
        # make predictions on the test set
        predictions = self.model.predict(self.x_test[self.features])
        
        # calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='binary')
        recall = recall_score(self.y_test, predictions, average='binary')
        f1 = f1_score(self.y_test, predictions, average='binary')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")