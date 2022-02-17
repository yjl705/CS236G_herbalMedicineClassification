# CS236G_herbalMedicineClassification
The repository is for course CS 236G use.
Our dataset is as follows: https://github.com/xzhan96-stf/Herbal-medicine-origin-e-nose.git


Classification with DRCA.py
is the file to train the LDA/SVM/RF model with DRCA and evaluate the model performance on the target train/target test set. The target dataset is partitioned into two datasets: target train and target test. The target train is used to train the DRCA domain adaptation model and optimize the hyperparameters of DRCA. The classifier's performance on the source domain data is used to feedback the hyperparameter tuning for the classifiers (LDA/SVM/RF). Once all the hyperparameters are tuned, the model performance is tested on the target domain test set.
