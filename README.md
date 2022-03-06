# CS236G_herbalMedicineClassification
The repository is for course CS 236G use.
Our first dataset (source domain data) cannot be shared due to the laboratory requirements. But the dataset has been introduced in the following publication: Zhan, X., Guan, X., Wu, R., Wang, Z., Wang, Y., & Li, G. (2018). Discrimination between alternative herbal medicines from different categories with the electronic nose. Sensors, 18(9), 2936.
Our second dataset is as follows: https://github.com/xzhan96-stf/Herbal-medicine-origin-e-nose.git
Dataset Description and the introduction to the features can be found in the link as well.


1. Classification with DRCA.py
is the file to train the LDA/SVM/RF model with DRCA and evaluate the model performance on the target train (target domain validation data in the report)/target test set. The target dataset is partitioned into two datasets: target train and target test. The target train is used to train the DRCA domain adaptation model and optimize the hyperparameters of DRCA. The classifier's performance on the source domain data is used to feedback the hyperparameter tuning for the classifiers (LDA/SVM/RF). Once all the hyperparameters are tuned, the model performance is tested on the target domain test set.

2. GAN_nn_agg.py
is the file to train the DNN model with DRCA and evaluate the model performance on the target train (target domain validation data in the report)/target test set. The partition of datasets are the same as file Classification with DRCA.py. The classifier's performance on the source domain data is used to feedback the hyperparameter tuning for the classifiers(DNN). Once all the hyperparameters are tuned, the model performance is tested on the target domain test set.

3. Classification with Cycle-GAN.py, Classification with DRCA.py, Classification with GAN_DRCA.py, Classification with GAN_KMM.py
are the files to train the LDA/SVM/RF models and evaluate the model performance on the target train set (target domain validation data in the report) and target domain test set. The hyperparameters of the classifiers (linear discrinant analysis, support vector machine and random forest) are tuned in a five-fold cross validation on the source domain data (as we deem the target domain data are without labels known to the users). The domain adaptation approaches' hyperparameters were manually tuned based on the performance on the target domain validation data.
