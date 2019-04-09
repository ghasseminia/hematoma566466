#!/usr/bin/python
import openpyxl as opxl
import numpy as np
import sys
import matplotlib.pyplot as plt
import optunity
import optunity.metrics
import sklearn.svm
 
 
def SimilarityMatrix(prediction,truth):
	matrix = [[0, 0],[0, 0]]
	for i in range(prediction.size):
		# true negative (predicted negative correctly)
		if prediction[i] == 0 and truth[i] == 0:
			matrix[1][1] += 1
		# false positive (predicted positive but is negative)
		elif prediction[i] == 1 and truth[i] == 0:
			matrix[0][1] += 1
		# true positive (predicted positive correctly)
		elif prediction[i] == 1 and truth[i] == 1:
			matrix[0][0] += 1
		# false negative (predicted negative but is positive)
		elif prediction[i] == 0 and truth[i] == 1:
			matrix[1][0] += 1
	print("         -----------------------")
	print("         --------True values----")
	print("------------Class 1----Class 0--")
	print("-P|-----------------------------")
	print("-r|---------   TP  ----   FP  --")
	print("-e|Class 1--   " + str(matrix[0][0]) + "   ----   "+ str(matrix[0][1])+ "   --")
	print("-d|-----------------------------")   
	print("-i|-----------------------------")
	print("-c|---------   FN  ----   TN  --")
	print("-t|Class 0--   " + str(matrix[1][0]) + "   ----   "+ str(matrix[1][1])+ "   --")
	print("------------------------------")
	#print("true negatives: ", matrix[1][1])
	#print("false negatives: ", matrix[1][0])
	#print("true positive: ", matrix[0][0])
	#print("false positives: ", matrix[0][1])
	print("\n")
	return matrix
	
def GetSplitData(size, datax,datay):
	trainx = np.matrix(datax[:size,:])
	testx = np.matrix(datax[size:,:])
	trainy = np.matrix(datay[:size])
	testy = np.matrix(datay[size:])
	return trainx, trainy, testx, testy 
	
	
	
def jaccardCoefficient(matrix):
	print("jaccard coefficient: " + str(float(matrix[0][0])/(matrix[0][1]+matrix[1][0]+matrix[0][0])))
	
def dice_coefficient(matrix):
	return 2.0*matrix[1][1]/((2*matrix[1][1])+(matrix[0][1])+(matrix[1][0]))
    #dice coefficient 2nt/(na + nb)
    #a_bigrams = set(a)
    #b_bigrams = set(b)
    #overlap = len(a_bigrams & b_bigrams)
    
    #print("dice: " +  str(overlap * 2.0/(len(a_bigrams) + len(b_bigrams))))"""
			
 
 
def computeBinary(difference,initial):	 
	m = len(difference)
	binary = []
	for i in range(m):
		# if 33% increase or 6ml increase we classify as growth
		if (difference[i] > 6) or (difference[i] > initial[i]*0.33):
			binary.append(1)
		else:
			binary.append(0)
	return binary		
		
def accuracy(binary):
	samples = float(len(binary))
	binary = np.absolute(binary)
	incorrect = np.sum(binary)
	correct = len(binary) - incorrect	
	accuracy = correct/samples
	return accuracy
		
def computeCost (X,y,theta):
    m = len(y)
    p1 = np.matmul(X,theta)-y
    p2 = p1.transpose()
    J=np.matmul(p2,p1)/(2*m)
        # print("J: ",J)
    return J
#############################################
def gradient_descent(X,y,theta,alpha,num_iters):
    m = len(y)
    J_history=np.zeros((num_iters,1))
    for i in range(0,num_iters):
        temp=np.matmul(X,theta)
        #print(temp, temp.shape)
        error=temp-y
        new_X=np.matmul(X.transpose(),error )
        theta_temp= ( (alpha/m)*new_X )
        #print(theta_temp)
        #print(theta_temp, "test")
        theta= theta-theta_temp;
    
        J_history[i][0]=computeCost (X,y,theta)
    
    if i!=0 and i!=1:
        if J_history[i][0]-J_history[i-1][0]< alpha: 
            #print("convergence criterion is satisfied")
            pass
    return(J_history,theta)
###################################################### 

       
np.set_printoptions(threshold=sys.maxsize,precision = 2,linewidth=200)
MAXROW = 76
SELECTED = ['E','D','F','G','H','K','L','N','Q','R','S','T','U','V','W','X','Y','Z', 'AA','AB','AC','AD','AE']
std_col = ['G','J','M','AI','AL','AO']
wb = opxl.load_workbook("linear_table_edited.xlsx")
data_set = wb['Sheet3']
std_sheet = wb['Sheet4']
X_matrix = []
for y in range(2,MAXROW+1):
    row = []
    if y!=8 and y!=13 and y!=23 and y!=32 and y!=44 and y!=50 and y!=53 and y!=55 and y!= 58 and y!= 67:
        for x in SELECTED:
            #print(x+str(y))
            cell_idx = x+str(y)
            # print(cell_idx)
            #print(data_set[cell_idx].value)
            row.append(float(data_set[cell_idx].value))
        # print(row)
        X_matrix.append(row)

X_matrix = np.matrix(X_matrix)
y_zero = []
initial_volumes = []
end_volumes = []
for y in range(2,MAXROW+1):
    if y!=8 and y!=13 and y!=23 and y!=32 and y!=44 and y!=50 and y!=53 and y!=55 and y!= 58 and y!= 67:
        cell_idx = 'M'+str(y)
        ICH_0hr = float(std_sheet[cell_idx].value)
        cell_idx = 'AO'+str(y)
        ICH_24hr = float(std_sheet[cell_idx].value)

        diff = ICH_24hr-(ICH_0hr)
        initial_volumes.append(ICH_0hr)
        end_volumes.append(ICH_24hr)
        y_zero.append(diff)
    
    
#print(X_matrix)
initial_volumes = np.matrix(initial_volumes)
initial_volumes = np.transpose(initial_volumes)
end_volumes = np.matrix(end_volumes)
end_volumes = np.transpose(end_volumes)
y_zero = np.matrix(y_zero)
y_zero = np.transpose(y_zero)
y_binary = computeBinary(y_zero,initial_volumes)
#print(y_binary)
# print(matrix.shape)

'''alpha=0.0000001
num_iters=10000'''
ones = np.full((65,1),1)
X_matrix = np.hstack((ones,X_matrix))
theta = np.full((23,1),1)
Cs = np.arange(100000.0)
Cs = Cs/100
min = 100
minC = 0
dice = 0
percent = 0
# split data, testSize and totalSize - testSize 
trainx, trainy, testx, testy = GetSplitData(50, X_matrix,np.array(y_binary))
trainy = np.array(trainy.tolist()[0])
testy = np.array(testy.tolist()[0])




# score function: twice iterated 5-fold cross-validated accuracy
@optunity.cross_validated(x=trainx, y=trainy, num_folds=4, num_iter=2)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
	model = sklearn.svm.SVC(C=10 ** logC, gamma=10 ** logGamma,kernel='rbf').fit(x_train, y_train)
	decision_values = model.decision_function(x_test)
	return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=50, logC=[-10, 10], logGamma=[-100, 100])
print(hps)

# train model on the full training set with tuned hyperparameters
optimal_model = sklearn.svm.SVC(C=3.1, gamma=0.00025).fit(X_matrix, y_binary)

output = []
ourguess = []
for j in range(15):
	test = optimal_model.predict(testx[j,:])
	ourguess.append(test[0])
			
ourguess = np.array(ourguess)
binary = []


for i in range(len(ourguess)):
	binary.append(ourguess[i]-testy[i])
percent = accuracy(binary)

simMatrix = SimilarityMatrix(ourguess,testy)
#print(simMatrix)
dice = dice_coefficient(simMatrix)



print("percent: " + str(percent))
print("Dice: " + str(dice))


