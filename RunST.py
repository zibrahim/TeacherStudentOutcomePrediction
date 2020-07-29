
import os
import json
import pandas as pd
import numpy as np

from Models.LSTM.Model import LSTMModel
from Models.XGBoost.Model import XGBoostModel
from Utils.Model import scale, stratified_group_k_fold, generate_trajectory_timeseries, impute

import shap
from sklearn import tree


def main():
    shap.initjs()
    ##1. read configuration file
    configs = json.load(open('Configuration.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    ##2. read data
    clustered_timeseries_path = configs['paths']['clustered_timeseries_path']
    time_series= pd.read_csv(clustered_timeseries_path+"TimeSeriesAggregatedClusteredDeltaTwoDays.csv")

    ##3. impute
    dynamic_features = configs['data']['dynamic_columns']
    grouping = configs['data']['grouping']
    time_series[dynamic_features] = impute(time_series, dynamic_features)

    ##4. generate new features based on delta from baseline
    outcome_columns = configs['data']['classification_outcome']
    baseline_features = configs['data']['baseline_columns']
    static_features = configs['data']['static_columns']

    new_series = generate_trajectory_timeseries(time_series, baseline_features, static_features,
                                                dynamic_features, grouping, outcome_columns)

    ##5. scale
    normalized_timeseries = scale(new_series, dynamic_features)

    groups = np.array(time_series[grouping])
    X = normalized_timeseries[dynamic_features]
    X_student = time_series[static_features]
    X_student[grouping] = time_series[grouping]

    print(" AFTER AGGREGATION, DIM OF X_STUDENT: ", X_student.shape)
    ##6. Training/Prediction for all outcomes.
    for outcome in configs['data']['classification_outcome']:

        outcome_df = pd.DataFrame()

        number_of_features = configs['data']['sequence_length']
        batch_size = configs['training']['batch_size']

        y = time_series[outcome]
        y = y.astype(int)
        teacher_model = LSTMModel(configs['model']['name'] + outcome)
        teacher_model.build_model(configs)

        student_model = XGBoostModel(configs['model']['name'] + outcome)

        for ffold_ind, (training_ind, testing_ind) in enumerate(
                stratified_group_k_fold(X, y, groups, k=5)) :  # CROSS-VALIDATION
            training_groups, testing_groups = groups[training_ind], groups[testing_ind]


            this_X_train, this_X_val = X.iloc[training_ind], X.iloc[testing_ind]
            this_y_train, this_y_val = y.iloc[training_ind], y.iloc[testing_ind]



            print("testing groups!!!!!", len(testing_groups), len(set(testing_groups)))
            this_y_ids = groups[testing_ind]

            assert len(set(training_groups) & set(testing_groups)) == 0

            #(NumberOfExamples, TimeSteps, FeaturesPerStep).

            reshaped_x = (this_X_train.values).reshape(-1, batch_size, number_of_features)
            reshaped_y = (this_y_train.values).reshape(-1, batch_size, 1)
            reshaped_x_val = (this_X_val.values).reshape(-1, batch_size, number_of_features)
            reshaped_y_val = (this_y_val.values).reshape(-1, batch_size,1)
            print(" THE RESHAPED: ")
            print(" TRAINING X SHAPE: ", reshaped_x.shape)
            print(" TRAINING Y SHAPE: ", reshaped_y.shape)
            print(" VAL X SHAPE: ", reshaped_x_val.shape)
            print(" VAL Y SHAPE: ", reshaped_y_val.shape)
            teacher_model.train(
                reshaped_x,
                reshaped_y,
                reshaped_x_val,
                reshaped_y_val,
                epochs=configs['training']['epochs'],
                batch_size=batch_size,
                save_dir=configs['model']['save_dir']
            )

            this_y_val = pd.DataFrame(this_y_val)
            this_y_val[grouping] = testing_groups
            print(" before reshaping:  ")
            print(" TRAINING X SHAPE: ", this_X_train.shape)
            print(" TRAINING Y SHAPE: ", this_y_train.shape)
            this_X_val.reset_index()

            y_pred_val_teacher = teacher_model.predict((this_X_val.values).reshape(-1,batch_size,number_of_features))
            print(" DIMENSIONS OF WHAT THE TEACHER PREDICTED: " ,y_pred_val_teacher.shape)

        ##ZI MAKE SURE YS CORRESPOND TO THE XS. DON'T JUST USE Y IN THIS CALL
        ## ZI WORK ON THIS

        print(" DIM OF Y PRED BY TEACHER:", y_pred_val_teacher.shape)
        print(" DIM OF THIS Y VAL: ", this_y_val.shape)

        #training_groups, testing_groups = groups[training_ind], groups[testing_ind]

        #this_X_train, this_X_val = X.iloc[training_ind], X.iloc[testing_ind]
        #this_y_train, this_y_val = y.iloc[training_ind], y.iloc[testing_ind]

        print(" COLUMNS OF THIS Y VAL WHICH IS XGBOOST TRAINING: ")
        print(this_y_val.columns)
        xgboost_y_training = this_y_val

        print(" PRINTING HEAD")
        print(xgboost_y_training.head())
        xgboost_y_training = xgboost_y_training.groupby(grouping).first()
        xgboost_y_training = xgboost_y_training.reset_index()
        lstm_output = pd.DataFrame(y_pred_val_teacher.reshape(len(xgboost_y_training), batch_size))
        lstm_output = lstm_output.reset_index()
        print(" SHAPES: df SO FAR: ", xgboost_y_training.shape, " LSTM OUTPUT: ", lstm_output.shape, type(lstm_output))
        xgboost_y_training = pd.merge(xgboost_y_training, lstm_output, left_index=True, right_index=True)

        #xgboost_y_training = pd.concat([xgboost_y_training, lstm_output], ignore_index=True, sort=False)
        #student_model.train(Xgboost_X, this_y_val, outcome, configs)


        static_df = time_series[static_features]
        static_df[grouping] = time_series[grouping]
        static_df = static_df.drop_duplicates(grouping)
        xgboost_y_training = xgboost_y_training.merge(static_df, how='left', on=grouping)
        xgboost_y_training.to_csv("StuentTrainig"+outcome+".csv")
        student_model.train(xgboost_y_training.iloc[:,3:], xgboost_y_training[outcome], outcome, configs)


    #plot_results(y_pred_val_binary, this_y_val)
if __name__ == '__main__':
    main()