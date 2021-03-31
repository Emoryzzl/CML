#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:11:04 2019
@author: zzl
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier #RandomForest
from sklearn.neural_network import MLPClassifier #CNN
from sklearn.svm import SVC #SVM  /Linear SVM, RBF SVM 
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #QDA
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve,auc
########## Prepare dataset
CML = np.load('CML_feature.npy').item()
Norm = np.load('Normal_feature.npy').item()

CML_case_num = len(CML['case_ids'])
Norm_case_num = len(Norm['case_ids'])

# props 1~4:mean,std,max,min
feature_name_list = ['mk_num','wbc_num','mk_density','mk_radious1','mk_radious2','mk_radious3','mk_radious4',
                     'voronoi_area1','voronoi_area2','voronoi_area3','voronoi_area4',
                     'voronoi_perimeter1','voronoi_perimeter2','voronoi_perimeter3','voronoi_perimeter4',
                     'delaunay_area1','delaunay_area2','delaunay_area3','delaunay_area4',
                     'delaunay_perimeter1','delaunay_perimeter2','delaunay_perimeter3','delaunay_perimeter4',
                     'lbp1','lbp2','lbp3','lbp4']

feature_num = len(feature_name_list)
CML_feature_matrix = np.zeros((CML_case_num,feature_num))
Norm_feature_matrix = np.zeros((Norm_case_num,feature_num))

for ii in range(CML_case_num):
    CML_feature_matrix[ii,0] = CML['mk_num'][ii]
    CML_feature_matrix[ii,1] = CML['wbc_num'][ii]
    CML_feature_matrix[ii,2] = CML['mk_density'][ii]
    CML_feature_matrix[ii,3:7] = CML['mk_radious'][ii]
    
    CML_feature_matrix[ii,7:15] = CML['voronoi_props'][ii]
    CML_feature_matrix[ii,15:23] = CML['delaunay_props'][ii]
    CML_feature_matrix[ii,23:27] = CML['lbp_props'][ii]

for jj in range(Norm_case_num):
    Norm_feature_matrix[jj,0] = Norm['mk_num'][jj]
    Norm_feature_matrix[jj,1] = Norm['wbc_num'][jj]
    Norm_feature_matrix[jj,2] = Norm['mk_density'][jj]
    Norm_feature_matrix[jj,3:7] = Norm['mk_radious'][jj]
    
    Norm_feature_matrix[jj,7:15] = Norm['voronoi_props'][jj]
    Norm_feature_matrix[jj,15:23] = Norm['delaunay_props'][jj]
    Norm_feature_matrix[jj,23:27] = Norm['lbp_props'][jj]

#Target
cml_label = np.ones((CML_case_num,1))
norm_label = np.zeros((Norm_case_num,1))

#Image feature
cml_dataset = np.column_stack((CML_feature_matrix,cml_label))
norm_dataset = np.column_stack((Norm_feature_matrix,norm_label))
fdataset = np.row_stack((cml_dataset,norm_dataset))

permutation = np.random.permutation(fdataset.shape[0])
original_shuffled_dataset = fdataset[permutation, :]

def t_test(dataset,feature_name_list,show_box_plot=False):
    ########## Feature boxplot
    from scipy import stats
    import pandas as pd

    df = pd.DataFrame(fdataset,columns = feature_name_list+['is_cml'])
    data = df.copy()
    num_feature = df.shape[1]-1
    p_value = []
    for kk in range(num_feature):
        feature_name = data.columns[kk]
        target_name = data.columns[-1]
        
        a = data[feature_name][data[target_name]==0]
        b = data[feature_name][data[target_name]==1]
        z = stats.ttest_ind(a, b)
        p_value.append(z.pvalue)
        if show_box_plot:
           plt.figure()
           plt.subplot(121)
           plt.boxplot(a, labels = ['{} & Norm'.format(feature_name)], sym = "bo")
           plt.subplot(122)
           plt.boxplot(b, labels = ['{} & CML'.format(feature_name)], sym = "ro")
           plt.suptitle('{}. cml .VS. Norm'.format(feature_name))
           plt.text(0,b.max()*1,'Pvalue={:.5e}'.format(z.pvalue),color='green')
           plt.show()
    return p_value


## Feature selection
p_value = t_test(fdataset,feature_name_list,show_box_plot=False)
selected_feature_index = np.array(p_value) < 0.01
selected_feature_num = selected_feature_index.sum()

shuffled_dataset = np.zeros((original_shuffled_dataset.shape[0],selected_feature_num+1))
fff = 0
selected_feature_name = []
for ff in range(feature_num):
    if selected_feature_index[ff]:
       shuffled_dataset[:,fff] = original_shuffled_dataset[:,ff]
       fff += 1
       selected_feature_name.append(feature_name_list[ff])
shuffled_dataset[:,-1] = original_shuffled_dataset[:,-1]        
########## Cross train & Independent test
cross_train = shuffled_dataset[:,:]  
independent_test = shuffled_dataset[:,:]        # 70:19
########## Classifier initial
lda = LinearDiscriminantAnalysis(n_components=1)
rfc = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=1,n_jobs=-1)
cnn = MLPClassifier(alpha=1, max_iter=1000)
Linear_SVM = SVC(kernel="linear", C=0.025,probability=True)
DecisionTree = DecisionTreeClassifier(max_depth=5)
AdaBoost = AdaBoostClassifier()
qda = QuadraticDiscriminantAnalysis()
knc = KNeighborsClassifier(3,n_jobs=-1)
########## Cross validation
A = cross_train[:,:-1]
y = cross_train[:,-1]
runs = 100
folders = 3
k_fold = sklearn.model_selection.KFold(folders, shuffle=True)

Avg_pos_prob = np.zeros((len(A),runs,8)) # For 80 cases
for run_time in range(runs):
    folder = 0
    for train_index, test_index in k_fold.split(A):
        s_time = time.time()
        A_train, A_test = A[train_index], A[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #Fitting in every classifier 
        lda.fit(A_train, y_train)
        rfc.fit(A_train, y_train)
        cnn.fit(A_train, y_train)
        Linear_SVM.fit(A_train, y_train)
        DecisionTree.fit(A_train, y_train)
        AdaBoost.fit(A_train, y_train)
        qda.fit(A_train, y_train)
        knc.fit(A_train, y_train)
        
        #Predicting proba in every classifier
        lda_proba = lda.predict_proba(A_test)
        rfc_proba = rfc.predict_proba(A_test)
        cnn_proba = cnn.predict_proba(A_test)
        Linear_SVM_proba = Linear_SVM.predict_proba(A_test)
        DecisionTree_proba = DecisionTree.predict_proba(A_test)
        AdaBoost_proba = AdaBoost.predict_proba(A_test)
        qda_proba = qda.predict_proba(A_test)
        knc_proba = knc.predict_proba(A_test)
        
        ##########              
        for zz in range(len(test_index)):
            Avg_pos_prob[test_index[zz],run_time,0] = lda_proba[zz][1] # 0 for negtive
            Avg_pos_prob[test_index[zz],run_time,1] = rfc_proba[zz][1]
            Avg_pos_prob[test_index[zz],run_time,2] = cnn_proba[zz][1]
            Avg_pos_prob[test_index[zz],run_time,3] = Linear_SVM_proba[zz][1]
            Avg_pos_prob[test_index[zz],run_time,4] = DecisionTree_proba[zz][1]
            Avg_pos_prob[test_index[zz],run_time,5] = AdaBoost_proba[zz][1]
            Avg_pos_prob[test_index[zz],run_time,6] = qda_proba[zz][1]
            Avg_pos_prob[test_index[zz],run_time,7] = knc_proba[zz][1]
        e_time = time.time() 
        print('{}/{} folder, {}/{} runs, taken:{} S'.format(folder+1,folders,run_time+1,runs,e_time-s_time))
        folder = folder+1 
    #print('{}/{} runs, taken:{} S'.format(run_time+1,runs,e_time-s_time))

np.save('CV_exp_result.npy',Avg_pos_prob)
np.save('CV_exp_lable.npy',y)

def AUC(y,prob):
    fpr, tpr, thresholds = roc_curve(y, prob)
    auc_value = auc(fpr, tpr)
    return auc_value,fpr, tpr
########### For selecting classifier
#eval_matrix = np.zeros((8,runs))
#eval_fresult = np.zeros((8,2))
#for ii in range(8):
#    for jj in range(runs):
#        prob = Avg_pos_prob[:,jj,ii]
#        auc_value,fpr, tpr = AUC(y,prob)
#        eval_matrix[ii,jj] = auc_value
#    eval_fresult[ii,0] = eval_matrix[ii,:].mean()
#    eval_fresult[ii,1] = eval_matrix[ii,:].std()

# For plotting
result = []
for kk in range(8):
    all_prob = Avg_pos_prob[:,:,kk]
    mean_prob = all_prob.mean(1)
    auc_value,fpr, tpr = AUC(y,mean_prob)
    result.append([auc_value,fpr, tpr])

## Plot ROC curve
plt.title('ROC Curve')
plt.plot(result[0][1], result[0][2], lw=1, label='LDA AUC={:.4f}'.format(result[0][0]))
plt.plot(result[1][1], result[1][2], lw=1, label='Random Forest AUC={:.4f}'.format(result[1][0]))
plt.plot(result[2][1], result[2][2], label='Neural Net AUC={:.4f}'.format(result[2][0]))
plt.plot(result[3][1], result[3][2], lw=1, label='Linear SVM AUC={:.4f}'.format(result[3][0]))
plt.plot(result[4][1], result[4][2], lw=1, label='Decision Tree AUC={:.4f}'.format(result[4][0]))
plt.plot(result[5][1], result[5][2], lw=1, label='AdaBoost AUC={:.4f}'.format(result[5][0]))
plt.plot(result[6][1], result[6][2], lw=1, label='QDA AUC={:.4f}'.format(result[6][0]))
plt.plot(result[7][1], result[7][2], lw=1, label='Nearest Neighbors AUC={:.4f}'.format(result[7][0]))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0,1)
plt.ylim(0,1)

plt.legend()    
plt.show()

########### Predicting on independent test set
#test_X = independent_test[:,:-1]
#test_Y = independent_test[:,-1]
#Linear_SVM.fit(A, y)    #lda,rfc,cnn,knc,Linear_SVM,DecisionTree,qda
#test_proba = lda.predict_proba(test_X)
#test_acc = lda.score(test_X, test_Y)
#print('Acc:{:.4f}'.format(test_acc))
#test_auc,test_fpr,test_tpr = AUC(test_Y, test_proba[:,1])
#   
### Plot ROC curve
#plt.title('Test ROC Curve')
#plt.plot(test_fpr, test_tpr, lw=1, label='AUC={:.4f}'.format(test_auc))
#plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.xlim(0,1)
#plt.ylim(0,1)
#
#plt.legend()
#plt.show()
















