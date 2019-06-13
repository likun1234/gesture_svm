# coding:utf-8
import numpy as np
import sklearn.svm as svm
from sklearn.externals import joblib
import pickle
from sklearn.model_selection import train_test_split,cross_val_score
from PIL import Image
import cv2
import os
class TSVM(object):
    def __init__(self):
        pass

    def initial(self, kernel='linear'):
        '''
        Initial TSVM
        Parameters
        ----------
        kernel: kernel of svm
        '''
        self.Cl, self.Cu = 1.5, 0.001
        self.kernel = kernel
        self.clf = svm.SVC(C=1.5, kernel=self.kernel)

    def load(self, model_path='./TSVM.model'):
        '''
        Load TSVM from model_path
        Parameters
        ----------
        model_path: model path of TSVM
                        model should be svm in sklearn and saved by sklearn.externals.joblib
        '''
        self.clf = joblib.load(model_path)

    def train(self, X1, Y1, X2):
        '''
        Train TSVM by X1, Y1, X2
        Parameters
        ----------
        X1: Input data with labels
                np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        Y1: labels of X1
                np.array, shape:[n1, ], n1: numbers of samples with labels
        X2: Input data without labels
                np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features
        '''
        N = len(X1) + len(X2)
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu

        self.clf.fit(X1, Y1)
        Y2 = self.clf.predict(X2)
        Y2 = np.expand_dims(Y2, 1)
        X2_id = np.arange(len(X2))
        X3 = np.vstack([X1, X2])
        Y3 = np.vstack([Y1, Y2])
        print (Y3)
        #while self.Cu < self.Cl:
        #    self.clf.fit(X3, Y3, sample_weight=sample_weight)
        #    while True:
        #        Y2_d = self.clf.decision_function(X2)    # linear: w^Tx + b
        #        Y2 = Y2.reshape(-1)
        #        epsilon = 1 - Y2 * Y2_d   # calculate function margin
        #        positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
        #        negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]
        #        positive_max_id = positive_id[np.argmax(positive_set)]
        #        negative_max_id = negative_id[np.argmax(negative_set)]
        #        a, b = epsilon[positive_max_id], epsilon[negative_max_id]
        #        if a > 0 and b > 0 and a + b > 2.0:
        #            Y2[positive_max_id] = Y2[positive_max_id] * -1
        #            Y2[negative_max_id] = Y2[negative_max_id] * -1
        #            Y2 = np.expand_dims(Y2, 1)
        #            Y3 = np.vstack([Y1, Y2])
        #            self.clf.fit(X3, Y3, sample_weight=sample_weight)
        #        else:
        #            break
        #    self.Cu = min(2*self.Cu, self.Cl)
        #    sample_weight[len(X1):] = self.Cu

    def score(self, X, Y):
        '''
        Calculate accuracy of TSVM by X, Y
        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Y: labels of X
                np.array, shape:[n, ], n: numbers of samples
        Returns
        -------
        Accuracy of TSVM
                float
        '''
        return self.clf.score(X, Y)

    def predict(self, X):
        '''
        Feed X and predict Y by TSVM
        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Returns
        -------
        labels of X
                np.array, shape:[n, ], n: numbers of samples
        '''
        return self.clf.predict(X)

    def save(self, path='./TSVM.model'):
        '''
        Save TSVM to model_path
        Parameters
        ----------
        model_path: model path of TSVM
                        model should be svm in sklearn
        '''
        joblib.dump(self.clf, path)

if __name__ == '__main__':
    model = TSVM()
    model.initial()
    list_ = []
    list1 = os.listdir("/home/lk/gesture/my_hand/0/")
    list2 = os.listdir("/home/lk/gesture/my_hand/2/")
    list3 = os.listdir("/home/lk/gesture/my_hand/5/")
    list_ = list1+list2+list3
    len_ = len(list1)+len(list2)+len(list3)
    mat = np.zeros([len_,3364])
    mat_test = np.zeros([5,3364])
    mat_label = np.zeros([len_,1])
    mat_test_label = np.zeros([5,1])
    index = 0
    index1 = 0
    #print(len(list_))
    #exit()
    for image in list1:
        img = Image.open("/home/lk/gesture/my_hand/0/" + image)
        img_ndarry = np.asarray(img, dtype = 'float32')
        img_ndarry = img_ndarry.reshape(-1,58*58)
        mat[index] = img_ndarry
        mat_label[index] = 0
        index = index+1
    
    for image in list2:
        img = Image.open("/home/lk/gesture/my_hand/2/" + image)
        img_ndarry = np.asarray(img, dtype = 'float32')
        img_ndarry = img_ndarry.reshape(-1,58*58)
        mat[index] = img_ndarry
        mat_label[index] = 2
        index = index+1
    
    for image in list3:
        img = Image.open("/home/lk/gesture/my_hand/5/" + image)
        img_ndarry = np.asarray(img, dtype = 'float32')
        img_ndarry = img_ndarry.reshape(-1,58*58)
        mat[index] = img_ndarry
        mat_label[index] = 5
        index = index+1
    
    model.train(mat, mat_label,mat_test)
    list4 = os.listdir("/home/lk/gesture/my_hand/5_/")
    for image in list4:
        str_ = "/home/lk/gesture/my_hand/5_/" + image
        img = Image.open(str_)
        img_ndarry = np.asarray(img, dtype = 'float32')
        img_ndarry = img_ndarry.reshape(-1,58*58)
        pic = cv2.imread(str_)
        test = model.predict(img_ndarry)
        print(str_ + ": " + str(int(test[0])))
        mat_test[index1] = img_ndarry
        mat_test_label[index1] = 5
        index1 = index1+1

    
    
    Y_hat = model.predict(mat_test)
    print(Y_hat)
    accuracy = model.score(mat_test,mat_test_label)
    print(accuracy)
    #model.save()

