import openpyxl as opxl
import numpy as np
import sys

def computeCost (X,y,theta):
    m=65
    p1 = np.matmul(X,theta)-y
    p2 = p1.transpose()
    J = np.matmul(p2,p1)/(2*m)
    print("J: ",J)
    return J



np.set_printoptions(threshold=sys.maxsize,precision = 2,linewidth=200)
MAXROW = 76
SELECTED = ['E','F','G','H','K','L','N','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AZ']
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
            row.append(float(data_set[cell_idx].value))
        # print(row)
        X_matrix.append(row)

X_matrix = np.matrix(X_matrix)
y_zero = []
for y in range(2,MAXROW+1):
    if y!=8 and y!=13 and y!=23 and y!=32 and y!=44 and y!=50 and y!=53 and y!=55 and y!= 58 and y!= 67:
        cell_idx = 'G'+str(y)
        IPH_0hr = float(std_sheet[cell_idx].value)
        cell_idx = 'J'+str(y)
        IVH_0hr = float(std_sheet[cell_idx].value)
        cell_idx = 'M'+str(y)
        ICH_0hr = float(std_sheet[cell_idx].value)
        
        cell_idx = 'AI'+str(y)
        IPH_24hr = float(std_sheet[cell_idx].value)
        cell_idx = 'AL'+str(y)
        IVH_24hr = float(std_sheet[cell_idx].value)
        cell_idx = 'AO'+str(y)
        ICH_24hr = float(std_sheet[cell_idx].value)

        diff = IPH_24hr+IVH_24hr+ICH_24hr-(IPH_0hr+IVH_0hr+ICH_0hr)
        y_zero.append(diff)
        # print("row: ",y)
        # print(diff)
        # cell_idx = 'G'+str(y)
        # print(float(std_sheet[cell_idx].value))
#print(matrix)
y_zero = np.matrix(y_zero)
y_zero = np.transpose(y_zero)
# print(matrix.shape)
print("Y dimension:",y_zero.shape)
# print(y_zero.shape)

theta = np.full((24,1),1)
print(theta.shape)

ones = np.full((65,1),1)
X_matrix = np.hstack((ones,X_matrix))
#print(X_matrix)

computeCost(X_matrix,y_zero,theta)

