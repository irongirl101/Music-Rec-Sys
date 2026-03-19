import pandas as pd 
import sympy 
import numpy as np 
import csv

def rref_from_csv(filename):
    try:
        df = pd.read_csv(filename,header =None)
    except: 
        print("Error Occured")
    
    matrix = df.to_numpy(dtype=float) # convert to data fram to numpy array - easier for matrix ops, float to avoid integer division
    matrix_s = sympy.Matrix(matrix)

    rref,pivots = matrix_s.rref()

    return rref,pivots

filename = "main_data.csv"
rref,pivots = rref_from_csv(filename)

if rref is not None: 
    print(rref)
    print(pivots)


    

