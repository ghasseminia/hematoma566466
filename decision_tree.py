import numpy as np 
import pandas as pd 
import sys,random
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.impute import SimpleImputer
np.set_printoptions(threshold=sys.maxsize)

#--import the data--#########################
def import_data(filename):
    data = pd.read_excel(filename)
    # print("data length:", len(data))
    # print("data shape:",data.shape)

    # print("dataset:", data.head())
    return data



#--Function used to determine X and Y depending on what purpose of result_type-- ########################
def set_parameter(data,result_type):
    data_np = data.to_numpy()
    if result_type == "mortality":
        X = data_np[:,3:]
        X = X.astype(float)
        Y = data_np[:,2]
        Y = Y.astype(int)
        # print("Y, ",Y)
        return X,Y
    elif result_type == "expansion":
        X = data_np[:,3:]
        X = X.astype(float)
        final_ICH = data_np[:,2]
        acute_total = data_np[:,3:4]
        acute_total = np.sum(acute_total,1)
        acute_total = acute_total*1.3
        Y = np.zeros(np.shape(final_ICH))
        for i in range(np.size(acute_total)):
            if final_ICH[i]>=acute_total[i]:
                Y[i] = 1
        # print("final, ",final_ICH)
        # print("acute_total, ",acute_total)
        # print("Y, ",Y)
        return X,Y


#--Function to split the dataset for cross validation-- #########################
def cross_validation_split(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = random.randint(1,50000)) 
    return X_train, X_test, Y_train, Y_test 


#--Function to perform training with giniIndex-- #################################
def train_gini(X_train, X_test, Y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini",max_depth = 4, min_samples_split = 17, min_samples_leaf = 17) 
    # print(clf_gini)
    # Performing training 
    clf_gini.fit(X_train,Y_train) 
    return clf_gini 

#--Function to perform training with entropy-- #################################
def train_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = 4, min_samples_split =17, min_samples_leaf = 17) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 

#--function to produce all the outcome that i need==#####################
def produce_outcome(X,Y,data,test_type,t,result_type):
    
  
    overall_acc = 0
    overall_defauly_acc = 0
    overall_false_positive = 0
    overall_false_negative = 0

    #the overall most impactful features
    feature_weight = np.zeros(np.shape(X)[1])

    for i in range(t):
        X_train, X_test, Y_train, Y_test = cross_validation_split(X,Y)
        
        if test_type == "gini":
            clf_obj = train_gini(X_train,X_test,Y_train)
        else:
            clf_obj = train_entropy(X_train,X_test,Y_train)
        # print(clf_obj.tree_)
        Y_pred = clf_obj.predict(X_test) 
        # print(Y_pred)
        # print(Y_test)

        #accuracy of prediction
        acc_single = accuracy_score(Y_test,Y_pred)*100
        overall_acc += acc_single

        #default accuracy when guessing the most common
        if result_type == "mortality":
            default_acc = 100-np.sum(Y_test)/np.size(Y_test)*100
            overall_defauly_acc+=default_acc
        elif result_type == "expansion":
            default_acc = 100-np.sum(Y_test)/np.size(Y_test)*100
            overall_defauly_acc+=default_acc
        
        #false positive and negative count
        false_positive = 0
        false_negative = 0
        for j in range(np.size(Y_pred)):
            if Y_pred[j] == 1 and Y_test[j] == 0:
                false_positive +=1
            elif Y_pred[j] == 0 and Y_test[j] == 1:
                false_negative +=1
        overall_false_positive+=false_positive
        overall_false_negative+=false_negative

        #the weight of each feature
        feature_weight += clf_obj.feature_importances_

        

    print("average accuracy: ",overall_acc/t)
    print("average accuracy if choosing the most common: ",overall_defauly_acc/t)
    print("for ",np.size(Y_test)," samples in test, there are on average ",overall_false_positive/t, "false_positives")
    print("for ",np.size(Y_test)," samples in test, there are on average ",overall_false_negative/t, "false_negatives")

    #print the most prominent features
    print("the top ten most important features are: ")
    top =  np.argsort(feature_weight)[-8:]
    top =  np.flip(top)
    if result_type == "mortality":
        for idx in top:
            print(data.columns[idx+2],feature_weight[idx]/t)
    elif result_type == "expansion":
        for idx in top:
            print(data.columns[idx+3],feature_weight[idx]/t)
#--main--###########################################
def main():
    # input data format requirement:
    # column0,1: study number, name initial
    random.seed()
    data = import_data(sys.argv[1])

    t = int(sys.argv[2])

    #parse X,Y
    try:
        X,Y = set_parameter(data,sys.argv[3])
    except:
        print("wrong input type")
    
    print("Gini: ")
    produce_outcome(X, Y,data,"gini",t,sys.argv[3])
    print("-----------------------------------------")
    print("Entropy: ")
    produce_outcome(X, Y,data,"entropy",t,sys.argv[3])
    
if __name__=="__main__": 
    main() 