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
import scipy
import pandas as pd
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

def transform(Xs, Xt, dim=30, kernel_type='primal'):
    '''
    Transform Xs and Xt
    :param Xs: ns * n_feature, source feature
    :param Xt: nt * n_feature, target feature
    :return: Xs_new and Xt_new
    '''
    kernel_type = 'primal'
    lamb = 1
    gamma = 1

    X = np.hstack((Xs.T, Xt.T))
    X /= np.linalg.norm(X, axis=0)
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)
    e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
    M = e * e.T
    M = M / np.linalg.norm(M, 'fro')
    H = np.eye(n) - 1 / n * np.ones((n, n))
    K = kernel('primal', X, None, gamma=gamma)
    n_eye = m if kernel_type == 'primal' else n
    a, b = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
    w, V = scipy.linalg.eig(a, b)
    ind = np.argsort(w)
    A = V[:, ind[:dim]]
    Z = np.dot(A.T, K)
    Z /= np.linalg.norm(Z, axis=0)
    Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
    return Xs_new, Xt_new

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

def paraSearch(baseModel, paraDict, X, Y):
    clf = GridSearchCV(baseModel, param_grid=paraDict, scoring='accuracy')
    search = clf.fit(X, Y)
    bestParaDict = search.best_params_
    bestModel = search.best_estimator_
    return bestParaDict, bestModel

from scipy import io
os.chdir(Data_X_dir)
source = io.loadmat('Source_BZ_DG_GG.mat')
Xs = np.row_stack((source['BZ'],source['DG'],source['GG']))
Ys = np.row_stack((np.zeros((50,1)),np.ones((50,1)),np.ones((50,1))+1))

target = io.loadmat('Target_BZ_DG_GG.mat')
Xt = np.row_stack((target['BZ'],target['DG'],target['GG']))
Yt = np.row_stack((np.zeros((160,1)),np.ones((160,1)),np.ones((160,1))+1))

train_idx = np.concatenate((np.arange(0,100),np.arange(160,260),np.arange(320,420))).reshape(-1,)
test_idx = np.concatenate((np.arange(100,160),np.arange(260,320),np.arange(420,480))).reshape(-1,)

Yt_train = Yt[train_idx,:]
Yt_test = Yt[test_idx,:]

os.chdir(Data_generated_dir)
GAN_model_name = 'NoiseAdding0.7_Cycle-GAN_S2T_three layers_nonsaturating_separateGDE_dim50_epoch2000_G initial_lr0.001_D initial_lr0.0001_Grec initial_lr0.005_beta1_alpha1'
Xt_train = np.load(GAN_model_name+'train.npy')
assert Xt_train.shape[0] == Yt_train.shape[0]
Xt_test = np.load(GAN_model_name+'test.npy')
assert Xt_test.shape[0] == Yt_test.shape[0]

Ds = [10,20,50,80,100]
kernels = ['primal','linear','rbf']

for D in Ds:
    for alpha in kernels:
        if True:
            #Standardization
            #Original Feature Space
            ss = StandardScaler()
            Xs_std = ss.fit_transform(Xs)
            Xt_train_std = ss.transform(Xt_train)
            Xt_test_std = ss.transform(Xt_test)

            #Projected Feature Space
            Xs_std_drca, Xt_std_drca = transform(Xs=Xs_std, Xt=np.concatenate((Xt_train_std,Xt_test_std),axis=0), dim=D, kernel_type=alpha)
            Xt_train_drca = Xt_std_drca[0:Xt_train_std.shape[0],:]
            Xt_test_drca = Xt_std_drca[Xt_train_std.shape[0]:, :]

            assert Xt_test_drca.shape[0] == Xt_test_std.shape[0]
            assert Xt_train_drca.shape[0] == Xt_train_std.shape[0]


            ### ---- Direct Prediction --- ###
            lda_base = LinearDiscriminantAnalysis(solver='lsqr')
            svm_base = SVC()
            knn_base = KNeighborsClassifier()
            rf_base = RandomForestClassifier()

            ### --- With Hyperparameter Tuning --- ###

            # model_list = [lda_base, svm_base, knn_base, rf_base]
            #model_name = ['lda', 'svm', 'knn', 'rf']
            # Parameters
            ldaPara = {'shrinkage': [10**i for i in np.random.uniform(low=-3, high=0, size=20)]}
            svmPara = {'C': [10**i for i in np.random.uniform(low=-3, high=2, size=20)], 'kernel': ['linear','rbf', 'sigmoid'],
                       'gamma': ['scale', 'auto']}
            knnPara = {'n_neighbors': np.arange(0,5)*2+1}
            rfPara = {'n_estimators': [50,100,200,300], 'max_depth': [4, 5, 6, 7, 8, 9, 10, 12, 15]}
            #para_list = [ldaPara, svmPara, knnPara, rfPara]

            model_list = [svm_base]
            model_name = ['svm']
            para_list = [svmPara]
            paraDict = {}
            bestModelDict = {}
            for i in range(len(model_list)):
                #print('Model tuning: ', model_name[i])
                bestParaDict, bestModel = paraSearch(model_list[i], para_list[i], Xs_std_drca, Ys)
                paraDict[model_name[i]] = bestParaDict
                bestModelDict[model_name[i]] = bestModel

            os.chdir(Dir_hyperparameters)
            joblib.dump(paraDict,filename='GANKMM_SVM_dim'+str(D)+'_alpha' + str(alpha)+'_best hyperparameters')

            ### ---- Direct Prediction --- ###
            results_dict = {}
            for i, (model_name,model) in enumerate(bestModelDict.items()):
                result_list = []
                results_dict[model_name] = {}
                model.fit(Xs_std_drca,Ys)

                Xts_train_pred = model.predict(Xt_train_drca)
                Xts_train_acc = accuracy_score(y_true=Yt_train, y_pred=Xts_train_pred)
                results_dict[model_name]['Xt_train_drca'] = Xts_train_acc
                result_list.append(str(round(Xts_train_acc,3)))

                Xts_test_pred = model.predict(Xt_test_drca)
                Xts_test_acc = accuracy_score(y_true=Yt_test, y_pred=Xts_test_pred)
                results_dict[model_name]['Xt_test_drca'] = Xts_test_acc
                result_list.append(str(round(Xts_test_acc,3)))

                print(';'.join([model_name,str(D),str(alpha)] + result_list))

            os.chdir(Dir_results)
            joblib.dump(results_dict,filename='GANKMM_SVM_dim'+str(D)+'_alpha' + str(alpha)+'_results')





