import config
from preprocess import Pipeline
import pandas as pd


pipeline = Pipeline(target=config.TARGET,
                    num_reciprocal=config.NUM_RECIPROCAL,
                    num_yeo_johnson=config.NUM_YEO_JOHNSON,
                    features = config.FEATURES)


if __name__ == "__main__":
    
    # load data
    data = pd.read_csv(config.DATASET_PATH)
    
    pipeline.fit(data)
    print("model Performance")
    pipeline.evaluate()
    print()

