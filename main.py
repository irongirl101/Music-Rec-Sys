import pandas as pd 
import sympy 
import numpy as np 
import csv
from scipy.linalg import qr 

def rref_from_csv(filename):
    try:
        df = pd.read_csv(filename,header =None)
    except: 
        print("Error Occured")
    
    matrix = df.to_numpy(dtype=float) # convert to data fram to numpy array - easier for matrix ops, float to avoid integer division
    matrix_s = sympy.Matrix(matrix)

    rref,pivots = matrix_s.rref()

    return rref,pivots,matrix_s

def spaces(rref,pivots,matrix_s): 
    rowspace_basis = []
    for i in range(rref.rows): 
        row = rref.row(i)
        if not row.is_zero_matrix: 
            rowspace_basis.append(list(row))
    colspace_basis = [] 
    for j in pivots: 
        col = matrix_s.col(j)
        colspace_basis.append(list(col))
    nullspace = matrix_s.nullspace()
    nullspace_basis = [list(i) for i in nullspace]

    return rowspace_basis,colspace_basis,nullspace_basis
    
def orthogonalize(vectors): 
    A = np.stack(vectors,axis = 1)
    Q,R = qr(A)
    return Q    


filename = "main_data.csv"
rref,pivots,matrix_s = rref_from_csv(filename)
#rref == rowspace 
rowspace  = rref
#if rref is not None: 
    #print(rref)
    #print(pivots)
rs,cs,ns = spaces(rref,pivots,matrix_s)
Q = orthogonalize(rs) # orthogonalized basis 
print(Q)

