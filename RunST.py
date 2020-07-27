
import os
import json
import pandas as pd
import numpy as np

from Models.LSTM.TeacherModel import LSTMTeacherModel
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
        teacher_model = LSTMTeacherModel(configs['model']['name'] + outcome)
        teacher_model.build_model(configs)

        student_model = XGBoostModel(configs['model']['name'] + outcome)

        for ffold_ind, (training_ind, testing_ind) in enumerate(
                stratified_group_k_fold(X, y, groups, k=5)) :  # CROSS-VALIDATION
            training_groups, testing_groups = groups[training_ind], groups[testing_ind]

            this_X_train, this_X_val = X.iloc[training_ind], X.iloc[testing_ind]
            this_y_train, this_y_val = y.iloc[training_ind], y.iloc[testing_ind]

            this_y_ids = groups[testing_ind]
            this_y_train = this_y_train.values.reshape(-1,batch_size)


            assert len(set(training_groups) & set(testing_groups)) == 0

            #(NumberOfExamples, TimeSteps, FeaturesPerStep).

            print(" TRAINING: ")
            print(" TRAINING X SHAPE: ", this_X_train.shape)
            print(" TRAINING Y SHAPE: ", this_y_train.shape)
            teacher_model.train(
                (this_X_train.values).reshape(-1, batch_size, number_of_features),
                (this_y_train).reshape(-1,batch_size,1),
                epochs=configs['training']['epochs'],
                batch_size=batch_size,
                save_dir=configs['model']['save_dir']
            )

            this_X_val.reset_index()

            y_pred_val_teacher = teacher_model.predict((this_X_val.values).reshape(-1,batch_size,number_of_features))


        ##ZI MAKE SURE YS CORRESPOND TO THE XS. DON'T JUST USE Y IN THIS CALL

        Xgboost_X = y_pred_val_teacher.reshape(len(this_y_val), 24)
        Xgboost_X = pd.DataFrame(Xgboost_X)

        blah = X_student.loc[X_student['PatientID'].isin(this_y_ids)]
        print("Outcomes/ClusteringTraining/X STUDENT SHAPE AFTER subsetting : ", blah.shape)

        X_student.to_csv("XStudent"+outcome+".csv")
        Xgboost_X[static_features] = blah[static_features]
        print(' BLAH SHAPE: ', blah.shape)
        student_model.train(Xgboost_X, this_y_val, outcome, configs)

    #plot_results(y_pred_val_binary, this_y_val)
if __name__ == '__main__':
    main()