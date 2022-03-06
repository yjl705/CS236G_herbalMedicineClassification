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

class DRCA():
    '''
    The DRCA Class
    '''

    def __init__(self, n_components=2, alpha=None, mode='raw'):
        '''
        The function to initialize the DRCA class
        :param n_components: The intended dimensionality of projection hyperplane smaller than the initial dimensionality
        :param alpha: weighting factor for target domain data within class scatter
        :param mode: the mode of DRCA:
            'raw': consider source domain data (S) and target domain data (T) as two groups
            'number': consider type-specific source domain data and target domain data based on the average number of cases in S and T
            'mean': equal weights for each class
        '''
        self.mode = mode
        self.Sw_s = None
        self.Sw_t = None
        self.mu_s = None
        self.mu_t = None
        self.alpha = alpha
        self.D_tilde = n_components

    pass

    def fit(self, Xs, Xt, Ys=None, Yt=None):
        '''
        This function fit the DRCA model with the data and labels given by users
        :param Xs: the feature matrix of shape (Ns, D) in source domain, np.array
        :param Xt: the feature matrix of shape (Nt, D) in target domain, np.array
        :param Ys: the label of the data of shape (Ns,) in source domain, np.array, int
        :param Yt: the label of the data of shape (Nt,) in target domain, np.array, int
        '''
        ### --- Summarize statistics --- ###
        if self.mode != 'raw':
            Ys = Ys.reshape(-1, )  # we need to use Y and make sure the Y is the intended form
            Yt = Yt.reshape(-1, )
        Ns = Xs.shape[0]
        Nt = Xt.shape[0]
        D = Xs.shape[1]

        ### --- Within-domain scatter --- ###
        self.mu_s = np.mean(Xs, axis=0, keepdims=True)  # 1*D
        self.mu_t = np.mean(Xt, axis=0, keepdims=True)
        self.Sw_s = (Xs - self.mu_s).T @ (Xs - self.mu_s)  # D*D
        self.Sw_t = (Xt - self.mu_t).T @ (Xt - self.mu_t)  # D*D
        if self.alpha == None:
            self.alpha = Ns / Nt
        self.nominator = self.Sw_s + self.Sw_t * self.alpha

        ### --- Eliminate sensor drifts --- ###
        if self.mode == 'raw':  # S and T as two entities
            self.denominator = (self.mu_s - self.mu_t).T @ (self.mu_s - self.mu_t)  # D*D
        elif self.mode == 'number':  # Focus on the same classes appeared in target domain
            Kt = np.unique(Yt).shape[0]  # Assume that the target domain classes are fewer
            self.denominator = np.empty((D, D))
            for i in range(Kt):
                Ns = np.mean(Ys == Kt[i])
                Nt = np.mean(Yt == Kt[i])
                N = 0.5 * (self.Ns + self.Nt)  # self. ???????????????????
                mu_s_matrix = np.mean(Xs[Ys == Kt[i], :], axis=0, keepdims=True)
                mu_t_matrix = np.mean(Xt[Yt == Kt[i], :], axis=0, keepdims=True)
                Sb_matrix = (self.mu_s_matrix - self.mu_t_matrix).T @ (self.mu_s_matrix - self.mu_t_matrix)
                self.denomiator += N * Sb_matrix
        elif self.mode == 'mean':  # Equal weights for every class
            Kt = np.unique(Yt).shape[0]  # Assume that the target domain classes are fewer
            self.denominator = np.empty((D, D))
            for i in range(Kt):
                mu_s_matrix = np.mean(Xs[Ys == Kt[i], :], axis=0, keepdims=True)  # 1*D
                mu_t_matrix = np.mean(Xt[Yt == Kt[i], :], axis=0, keepdims=True)  # 1*D
                Sb_matrix = (self.mu_s_matrix - self.mu_t_matrix).T @ (self.mu_s_matrix - self.mu_t_matrix)
                self.denomiator += Sb_matrix  # D*D

        eigenValues, eigenVectors = np.linalg.eig(np.linalg.pinv(self.denominator) @ self.nominator)  # D*D

        idx = np.abs(eigenValues).argsort()[::-1]
        self.eigenValues = eigenValues[idx]
        self.eigenVectors = eigenVectors[:, idx]
        self.W = self.eigenVectors[:, 0:self.D_tilde]  # shape=(D,D_tilde)

    pass

    def transform(self, X):
        '''
        This function use the fitted SRLDA model
        :param X: the data in np.array of shape (N,D) that needs to be projected to the lower dimension
        :return: X_tilde: the projected data in the lower dimensional space in np.array of shape (N, D_tilde)
        '''
        return np.real(np.matmul(X, self.W))  # goal:  (N,D_tilde)      (D_tilde*D)@(D*N).T     (N*D)(D*D_tilde)

    pass

    def fit_transform(self, Xs, Xt, Ys=None, Yt=None):
        '''
        :param Xs: the feature matrix of shape (Ns, D) in source domain, np.array
        :param Xt: the feature matrix of shape (Nt, D) in target domain, np.array
        :param Ys: the label of the data of shape (Ns,) in source domain, np.array, int
        :param Yt: the label of the data of shape (Nt,) in target domain, np.array, int '''

        self.fit(Xs, Xt, Ys, Yt)
        return np.real(self.transform(Xs)), np.real(self.transform(Xt))  # N * D_tilde

    pass

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

Xt_train = Xt[train_idx,:]
Yt_train = Yt[train_idx,:]
Xt_test = Xt[test_idx,:]
Yt_test = Yt[test_idx,:]

Ds = [10,20,50,80,100]
alphas = [10**i for i in np.random.uniform(low=-3, high=3, size=20)]

for D in Ds:
    for alpha in alphas:
        if True:
            #Standardization
            #Original Feature Space
            ss = StandardScaler()
            Xs_std = ss.fit_transform(Xs)
            Xt_train_std = ss.transform(Xt_train)
            Xt_test_std = ss.transform(Xt_test)

            #Projected Feature Space
            drca = DRCA(n_components=D, alpha=alpha)
            Xs_std_drca, Xt_train_drca = drca.fit_transform(Xs=Xs_std, Xt=Xt_train_std,Ys=Ys, Yt=Yt_train)
            Xt_test_drca = drca.transform(Xt_test_std)

            assert Xt_test_drca.shape[0] == Xt_test_std.shape[0]
            assert Xt_train_drca.shape[0] == Xt_train_std.shape[0]


            ### ---- Direct Prediction --- ###
            lda_base = LinearDiscriminantAnalysis(solver='lsqr')
            svm_base = SVC()
            knn_base = KNeighborsClassifier()
            rf_base = RandomForestClassifier()

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
                bestParaDict, bestModel = paraSearch(model_list[i], para_list[i], Xs_std_drca, Ys)
                paraDict[model_name[i]] = bestParaDict
                bestModelDict[model_name[i]] = bestModel

            os.chdir(Dir_hyperparameters)
            joblib.dump(paraDict,filename='DRCA_dim'+str(D)+'_alpha' + str(alpha)+'_best hyperparameters')

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
            joblib.dump(results_dict,filename='DRCA_dim'+str(D)+'_alpha' + str(alpha)+'_results')





