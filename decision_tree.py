import numpy as np 
import pandas as pd 
import sys,random,graphviz
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.impute import SimpleImputer
np.set_printoptions(threshold=sys.maxsize)

#--import the data--#########################
def import_data(filename):
    data = pd.read_excel(filename,sheet_name = "Sheet2")
    # print("data length:", len(data))
    # print("data shape:",data.shape)

    # print("dataset:", data.head())
    return data

def impute_data(X):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imp.fit_transform(X)
    # print(X)
    return X

#--Function used to determine X and Y depending on what purpose of result_type-- ########################
def set_parameter(data,result_type):
    data_np = data.to_numpy()
    if result_type == "mortality":
        X = data_np[:,3:]
        X = X.astype(float)
        X = impute_data(X)
        Y = data_np[:,2]
        Y = Y.astype(int)
        # print("Y, ",Y)
        return X,Y
    elif result_type == "expansion":
        X = data_np[:,3:]
        X = X.astype(float)
        X = impute_data(X)
        final_ICH = data_np[:,2]
        acute_total = data_np[:,3:5]
        # print(acute_total)
        acute_total = np.sum(acute_total,1)
        # print(acute_total)
        acute_total = acute_total*1.3
        # print(acute_total)
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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = random.randint(1,50000000)) 
    return X_train, X_test, Y_train, Y_test 


#--Function to perform training with giniIndex-- #################################
def train_gini(X_train, Y_train,md,msp,msl,mlf): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini",max_depth = md, min_samples_split = msp, min_samples_leaf = msl,max_leaf_nodes=mlf) 
    # print(clf_gini)
    # Performing training 
    clf_gini.fit(X_train,Y_train) 
    return clf_gini 

#--Function to perform training with entropy-- #################################
def train_entropy(X_train, y_train,md,msp,msl,mlf): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = md, min_samples_split =msp, min_samples_leaf = msl,max_leaf_nodes=mlf) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 

#--function to produce all the outcome that i need==#####################
def produce_outcome(X,Y,data,test_type,t,result_type,md,msp,msl,mlf):
    
  
    # comparison = []
    # for md in range(3,12):
    #     for msp in range(3,16):
    #         for msl in range(3,16):
    #             print (md,msp,msl)
    overall_acc = 0
    overall_default_acc = 0
    overall_false_positive = 0
    overall_false_negative = 0
    overall_correct_positive = 0
    overall_correct_negative = 0
    #the overall most impactful features
    feature_weight = np.zeros(np.shape(X)[1])

    largest_correct_positive = 0
    champion_false_negative = 20
    champion_false_negative = 20
    chosen_clf = None
    chosen_Y_pred = None
    chosen_Y_pred = None
    for i in range(t):
        X_train, X_test, Y_train, Y_test = cross_validation_split(X,Y)
        
        if test_type == "gini":
            clf_obj = train_gini(X_train,Y_train,md,msp,msl,mlf)
        else:
            clf_obj = train_entropy(X_train,Y_train,md,msp,msl,mlf)
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
            overall_default_acc+=default_acc
        elif result_type == "expansion":
            default_acc = 100-np.sum(Y_test)/np.size(Y_test)*100
            overall_default_acc+=default_acc
        
        #false positive and negative count
        false_positive = 0
        false_negative = 0
        correct_positive = 0
        correct_negative = 0
        for j in range(np.size(Y_pred)):
            if Y_pred[j] == 1 and Y_test[j] == 0:
                false_positive +=1
            elif Y_pred[j] == 0 and Y_test[j] == 1:
                false_negative +=1
            elif Y_pred[j] == 1 and Y_test[j] == 1:
                correct_positive +=1
            elif Y_pred[j] == 0 and Y_test[j] == 0:
                correct_negative += 1
        overall_false_positive+=false_positive
        overall_false_negative+=false_negative
        overall_correct_positive += correct_positive
        overall_correct_negative += correct_negative

        #the weight of each feature
        feature_weight += clf_obj.feature_importances_

        #chose the model with the largest correct_positive and the smallest false_negative
        if correct_positive > largest_correct_positive:
            largest_correct_positive = correct_positive
            champion_false_negative = false_negative
            champion_false_positive = false_positive
            chosen_clf = clf_obj
            chosen_Y_pred = Y_pred
            chosen_Y_test = Y_test
        elif correct_positive == largest_correct_positive:
            if false_negative < champion_false_negative:
                largest_correct_positive = correct_positive
                champion_false_negative = false_negative
                champion_false_positive = false_positive
                chosen_clf = clf_obj
                chosen_Y_pred = Y_pred
                chosen_Y_test = Y_test
            elif false_negative == champion_false_negative:
                if false_positive <= champion_false_positive:
                    largest_correct_positive = correct_positive
                    champion_false_negative = false_negative
                    champion_false_positive = false_positive
                    chosen_clf = clf_obj
                    chosen_Y_pred = Y_pred
                    chosen_Y_test = Y_test

    print("average accuracy: ",overall_acc/t)
    print("average accuracy if choosing the most common: ",overall_default_acc/t)
    print("for ",np.size(Y_test)," samples in test, there are on average ",overall_false_positive/t, "false_positives")
    print("for ",np.size(Y_test)," samples in test, there are on average ",overall_false_negative/t, "false_negatives")
    print("for ",np.size(Y_test)," samples in test, there are on average ",overall_correct_positive/t, "correct_positives")
    print("for ",np.size(Y_test)," samples in test, there are on average ",overall_correct_negative/t, "correct_negatives")
    print("The prediction from the best performing model is: ",chosen_Y_pred)
    print("The test result from the best performing model is:",chosen_Y_test)

    #print the most prominent features
    print("the top ten most important features are: ")
    top =  np.argsort(feature_weight)[-8:]
    top =  np.flip(top)
    if result_type == "mortality":
        for idx in top:
            print("-",data.columns[idx+3],feature_weight[idx]/t)
    elif result_type == "expansion":
        for idx in top:
            print("-",data.columns[idx+3],feature_weight[idx]/t)


    #export graph
    if result_type == "mortality":
        feature_name = np.array(data.columns[3:])
        test_name = result_type+"_"+test_type
        class_name = ["dead","alive"]
        visualize(chosen_clf,feature_name,class_name,test_name)
    elif result_type == "expansion":
        feature_name = np.array(data.columns[3:])
        test_name = result_type+"_"+test_type
        class_name = ["no expansion","expansion"]
        visualize(chosen_clf,feature_name,class_name,test_name)
    # print(feature_name)
        
    
    return overall_acc/t,overall_default_acc/t
    

#--function to search for the best parameter ==#####################
def find_parameters(X,Y,data,test_type,t,result_type):
    
    #calculate parameters for gini
    comparison = []
    for md in range(3,12):
        for msp in range(3,16):
            for msl in range(3,16):
                mlf = None
                # print (md,msp,msl)
                calculated_acc,default_acc = produce_outcome(X,Y,data,test_type,t,result_type,md,msp,msl,mlf)
                comparison.append([md,msp,msl,mlf,calculated_acc,default_acc])
        
    only_acc = np.array(comparison)[:,4]
    # print(only_acc)
    only_acc_idx = np.argsort(only_acc)
    only_acc_idx = np.flip(only_acc_idx)
    for idx in only_acc_idx:
        print(comparison[idx])    

def visualize(clf_obj,feature_name,class_name,test_name):
    dot_data = export_graphviz(clf_obj, out_file=None, feature_names=feature_name,  filled=True, rounded=True,  special_characters=True, class_names = class_name)
    # print(dot_data)
    graph = graphviz.Source(dot_data)  
    graph.render(test_name)
    pass

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
    # find_parameters(X, Y,data,"gini",t,sys.argv[3])
    produce_outcome(X, Y,data,"gini",t,sys.argv[3],10,15,4,None)
    print("-----------------------------------------")
    print("Entropy: ")
    # find_parameters(X, Y,data,"entropy",t,sys.argv[3])
    produce_outcome(X, Y,data,"entropy",t,sys.argv[3],3,6,4,None)

    
if __name__=="__main__": 
    main() 