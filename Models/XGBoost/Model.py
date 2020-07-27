import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import xgboost as xgb

import matplotlib.pyplot as plt



from Utils.Model import get_distribution
from sklearn.model_selection import KFold
class XGBoostModel() :
    def __init__( self , name):

        self.model = xgb.XGBClassifier(scale_pos_weight=263/73,
                           learning_rate=0.007,
                           n_estimators=100,
                           gamma=0,
                           max_depth=4,
                           min_child_weight=2,
                           subsample=1,
                           eval_metric='error')


    def train(self, X,y, label, configs):
        #X = X.reshape(len(y),24)
        #X = pd.DataFrame(X)
        y = y.astype(int)
        X.reset_index()
        y.reset_index()
        distrs = [get_distribution(y)]
        index = ['Entire set']

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 10) #CROSS VALIDATION CHANGE
        plt.figure(figsize=(10, 10))

        outcome_df = pd.DataFrame()

        kf = KFold(n_splits=3)


        print(" X LENGTH: ", len(X), " Y LENGTH: ", len(y))
        for train_index, test_index in kf.split(X) :
            training_X, testing_X = X.iloc[train_index], X.iloc[test_index]
            training_y, testing_y = y.iloc[train_index], y.iloc[test_index]

        # Train, predict and Plot
            self.model.fit(training_X,training_y)
            y_pred_rt = self.model.predict_proba(testing_X)[:, 1]

            y_pred_binary= (y_pred_rt > 0.5).astype('int32')

            F1Macro = f1_score(testing_y, y_pred_binary, average='macro')

            F1Micro = f1_score(testing_y, y_pred_binary, average='micro')
            F1Weighted = f1_score(testing_y, y_pred_binary, average='weighted')
            PrecisionMacro = precision_score(testing_y, y_pred_binary, average='macro')

            PrecisionMicro = precision_score(testing_y, y_pred_binary, average='micro')
            PrecisionWeighted = precision_score(testing_y, y_pred_binary, average='weighted')
            RecallMacro = recall_score(testing_y, y_pred_binary, average='macro')

            RecallMicro = recall_score(testing_y, y_pred_binary, average='micro')
            RecallWeighted = recall_score(testing_y, y_pred_binary, average='weighted')


            performance_row = {
                "F1-Macro" : F1Macro,
                "F1-Micro" : F1Micro,
                "F1-Weighted" : F1Weighted,
                "Precision-Macro" : PrecisionMacro,
                "Precision-Micro" : PrecisionMicro,
                "Precision-Weighted" : PrecisionWeighted,
                "Recall-Macro" : RecallMacro,
                "Recall-Micro" : RecallMicro,
                "Recall-Weighted" : RecallWeighted
            }

            outcome_df = outcome_df.append(performance_row, ignore_index=True)


            fpr, tpr, thresholds = roc_curve(testing_y, y_pred_rt)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold (AUC = %0.2f)' % ( roc_auc))

        # add to the distribution dataframe, for verification purposes
            distrs.append(get_distribution(training_y))

            index.append(f'training set - fold')
            distrs.append(get_distribution(testing_y))
            index.append(f'testing set - fold')

    #Shape plot
    # Finallise ROC curve
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)


        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.title('XGBoost Cross-Validation ROC', fontsize=18)
    # plt.legend(loc="lower right", prop={'size' : 15})


        plt.savefig(configs['model']['save_dir'] + label + "ROC_XGBoost.pdf")
        outcome_df.to_csv("Outcomes/"+label+"Student.csv")

        distr_df = pd.DataFrame(distrs, index=index, columns=[f'Label {l}' for l in range(np.max(y) + 1)])
        distr_df.to_csv(configs['model']['save_dir'] +"-K-Fold-Distributions.csv", index=True)
