import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor

from Utils.Model import get_distribution
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class XGBoostRegressionModel() :
    def __init__( self , name):
        self.model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001, random_state=0)

    def train(self, X,y, label, configs):

        X.reset_index()
        y.reset_index()
        distrs = [get_distribution(y)]
        index = ['Entire set']

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 10) #CROSS VALIDATION CHANGE
        plt.figure(figsize=(10, 10))

        outcome_df = pd.DataFrame()

        kf = KFold(n_splits=5)

        for train_index, test_index in kf.split(X) :
            training_X, testing_X = X.iloc[train_index], X.iloc[test_index]
            training_y, testing_y = y.iloc[train_index], y.iloc[test_index]

        # Train, predict and Plot
            self.model.fit(training_X,training_y)
            #y_pred_rt = self.model.predict_proba(testing_X)[:, 1]
            y_pred_rt = self.model.predict(testing_X)

            mse = mean_squared_error(testing_y, y_pred_rt) ** (0.5)

            performance_row = {
                "Mean Square Error" : mse
            }

            outcome_df = outcome_df.append(performance_row, ignore_index=True)
        outcome_df.to_csv("Outcomes/"+label+"RegressionStudent.csv")

        distr_df = pd.DataFrame(distrs, index=index, columns=[f'Label {l}' for l in range(np.max(y) + 1)])
        distr_df.to_csv(configs['model']['save_dir'] +"-K-Fold-Distributions.csv", index=True)
