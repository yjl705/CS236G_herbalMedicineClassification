from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
import warnings
from sklearn.exceptions import DataConversionWarning
import joblib
import pandas as pd
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

Data_X_dir = 'G:/我的云端硬盘/Paper/DRCA for Enose/Aggregated_feature'
Data_generated_dir = 'G:/我的云端硬盘/Paper/DRCA for Enose/GEN'
Dir_models = 'G:/我的云端硬盘/Paper/DRCA for Enose/GAN models'
Dir_results = 'G:/我的云端硬盘/Paper/DRCA for Enose/Results'
Dir_hyperparameters = 'G:/我的云端硬盘/Paper/DRCA for Enose/Hyperparameters'
try:
    os.mkdir(Dir_hyperparameters)
    os.mkdir(Dir_results)
except:
    print('Results directory already created!')

# The code to perform five-fold cross validation for hyperparameter tuning
def paraSearch(baseModel, paraDict, X, Y):
    clf = GridSearchCV(baseModel, param_grid=paraDict, scoring='accuracy')
    search = clf.fit(X, Y)
    bestParaDict = search.best_params_
    bestModel = search.best_estimator_
    return bestParaDict, bestModel

from scipy import io
#Load the data
os.chdir(Data_X_dir)
source = io.loadmat('Source_BZ_DG_GG.mat')
Xs = np.row_stack((source['BZ'],source['DG'],source['GG']))
Ys = np.row_stack((np.zeros((50,1)),np.ones((50,1)),np.ones((50,1))+1))

target = io.loadmat('Target_BZ_DG_GG.mat')
Xt = np.row_stack((target['BZ'],target['DG'],target['GG']))
Yt = np.row_stack((np.zeros((160,1)),np.ones((160,1)),np.ones((160,1))+1))

#Dataset partition
train_idx = np.concatenate((np.arange(0,100),np.arange(160,260),np.arange(320,420))).reshape(-1,)
test_idx = np.concatenate((np.arange(100,160),np.arange(260,320),np.arange(420,480))).reshape(-1,)

Xt_train = Xt[train_idx,:]
Yt_train = Yt[train_idx,:]
Xt_test = Xt[test_idx,:]
Yt_test = Yt[test_idx,:]


os.chdir(Data_generated_dir)
for root, dir, files in os.walk('.',topdown=False):
    for name in files:
        if ('NoiseAdding' in name) and ('test' in name) and ('0.2' not in name):
            os.chdir(Data_generated_dir)
            print('GAN file: ',name)
            Xts_test = np.load(name)
            assert Xts_test.shape[0] == Xt_test.shape[0]
            Xts_train = np.load(name[:-8]+'train.npy')
            assert Xts_train.shape[0] == Xt_train.shape[0]

            ss = StandardScaler()
            Xs_std = ss.fit_transform(Xs)
            Xt_train_std = ss.transform(Xt_train)
            Xt_test_std = ss.transform(Xt_test)
            Xts_train_std = ss.transform(Xts_train)
            Xts_test_std = ss.transform(Xts_test)

            ### ---- Direct Prediction --- ###
            lda_base = LinearDiscriminantAnalysis(solver='lsqr')
            svm_base = SVC()
            knn_base = KNeighborsClassifier()
            rf_base = RandomForestClassifier()

            lda_base.fit(Xs_std,Ys)
            svm_base.fit(Xs_std, Ys)
            knn_base.fit(Xs_std, Ys)
            rf_base.fit(Xs_std, Ys)

            #Baseline Prediction
            lda_pred = lda_base.predict(Xt_std)
            svm_pred = svm_base.predict(Xt_std)
            knn_pred = knn_base.predict(Xt_std)
            rf_pred = rf_base.predict(Xt_std)

            lda_acc = accuracy_score(y_true = Yt, y_pred=lda_pred)
            svm_acc = accuracy_score(y_true=Yt, y_pred=svm_pred)
            knn_acc = accuracy_score(y_true=Yt, y_pred=knn_pred)
            rf_acc = accuracy_score(y_true=Yt, y_pred=rf_pred)
            print(';'.join([str(lda_acc),str(svm_acc),str(knn_acc),str(rf_acc)]))

            #GAN Prediction
            lda_pred_gan = lda_base.predict(Xts_std)
            svm_pred_gan = svm_base.predict(Xts_std)
            knn_pred_gan = knn_base.predict(Xts_std)
            rf_pred_gan = rf_base.predict(Xts_std)

            lda_acc_gan = accuracy_score(y_true=Yt, y_pred=lda_pred_gan)
            svm_acc_gan = accuracy_score(y_true=Yt, y_pred=svm_pred_gan)
            knn_acc_gan = accuracy_score(y_true=Yt, y_pred=knn_pred_gan)
            rf_acc_gan = accuracy_score(y_true=Yt, y_pred=rf_pred_gan)
            print(';'.join([str(lda_acc_gan), str(svm_acc_gan), str(knn_acc_gan), str(rf_acc_gan)]))

            ### --- With Hyperparameter Tuning --- ###
            model_list = [lda_base, svm_base, knn_base, rf_base]
            model_name = ['lda', 'svm', 'knn', 'rf']
            # Parameters
            ldaPara = {'shrinkage': [10**i for i in np.random.uniform(low=-3, high=0, size=20)]}
            svmPara = {'C': [10**i for i in np.random.uniform(low=-3, high=2, size=20)], 'kernel': ['linear','rbf', 'sigmoid'],
                       'gamma': ['scale', 'auto']}
            knnPara = {'n_neighbors': np.arange(0,5)*2+1}
            rfPara = {'n_estimators': [50,100,200,300], 'max_depth': [4, 5, 6, 7, 8, 9, 10, 12, 15]}
            para_list = [ldaPara, svmPara, knnPara, rfPara]
            paraDict = {}
            bestModelDict = {}
            for i in range(len(model_list)):
                #print('Model tuning: ', model_name[i])
                bestParaDict, bestModel = paraSearch(model_list[i], para_list[i], Xs_std, Ys)
                paraDict[model_name[i]] = bestParaDict
                bestModelDict[model_name[i]] = bestModel

            os.chdir(Dir_hyperparameters)
            joblib.dump(paraDict,filename=name[:-8]+'_std_best hyperparameters')

            ### ---- Direct Prediction --- ###
            results_dict = {}
            print(name[:-8])
            for i, (model_name,model) in enumerate(bestModelDict.items()):
                result_list = []
                results_dict[model_name] = {}
                model.fit(Xs_std,Ys)
                Xt_train_pred = model.predict(Xt_train_std)
                Xt_train_acc = accuracy_score(y_true=Yt_train, y_pred=Xt_train_pred)
                results_dict[model_name]['Xt_train_acc'] = Xt_train_acc
                result_list.append(str(round(Xt_train_acc,3)))

                Xts_train_pred = model.predict(Xts_train_std)
                Xts_train_acc = accuracy_score(y_true=Yt_train, y_pred=Xts_train_pred)
                results_dict[model_name]['Xts_train_acc'] = Xts_train_acc
                result_list.append(str(round(Xts_train_acc,3)))

                Xt_test_pred = model.predict(Xt_test_std)
                Xt_test_acc = accuracy_score(y_true=Yt_test, y_pred=Xt_test_pred)
                results_dict[model_name]['Xt_test_acc'] = Xt_test_acc
                result_list.append(str(round(Xt_test_acc,3)))

                Xts_test_pred = model.predict(Xts_test_std)
                Xts_test_acc = accuracy_score(y_true=Yt_test, y_pred=Xts_test_pred)
                results_dict[model_name]['Xts_test_acc'] = Xts_test_acc
                result_list.append(str(round(Xts_test_acc,3)))

                print(';'.join([model_name] + result_list))

            os.chdir(Dir_results)
            joblib.dump(results_dict,filename=name[:-8]+'_std_results')





