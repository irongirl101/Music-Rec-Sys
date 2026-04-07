import pandas as pd 
import sympy 
import numpy as np 
import csv
from scipy.linalg import qr 

print('''
Now Launching 
 /$$$$$$$$ /$$                               /$$$$$$$                        /$$   
| $$_____/|__/                              | $$__  $$                      | $$    
| $$       /$$  /$$$$$$   /$$$$$$  /$$$$$$$ | $$  \ $$  /$$$$$$   /$$$$$$  /$$$$$$  
| $$$$$   | $$ /$$__  $$ /$$__  $$| $$__  $$| $$$$$$$  /$$__  $$ |____  $$|_  $$_/  
| $$__/   | $$| $$  \ $$| $$$$$$$$| $$  \ $$| $$__  $$| $$$$$$$$  /$$$$$$$  | $$    
| $$      | $$| $$  | $$| $$_____/| $$  | $$| $$  \ $$| $$_____/ /$$__  $$  | $$ /$$
| $$$$$$$$| $$|  $$$$$$$|  $$$$$$$| $$  | $$| $$$$$$$/|  $$$$$$$|  $$$$$$$  |  $$$$/
|________/|__/ \____  $$ \_______/|__/  |__/|_______/  \_______/ \_______/   \___/  
               /$$  \ $$                                                           
              |  $$$$$$/                                                            
               \______/                                                             
''')

def rref_from_csv(filename):
    try:
        df = pd.read_csv(filename,header =None)
        numeric_df = df.drop(columns=[df.columns[0]]) 
        matrix = numeric_df.to_numpy(dtype=float) # convert to data fram to numpy array - easier for matrix ops, float to avoid integer division
        matrix_s = sympy.Matrix(matrix)
        matrix_s = sympy.Matrix(matrix)
    except Exception as e: 
        print(f"Error Occurred: {e}")
        return None, None, None
   

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


def diagonalize(matrix_s):
    
    A = matrix_s
    covariance_matrix = A.T * A 
    
    P, D = covariance_matrix.diagonalize()
    return P, D

def predict_recommendation(matrix_s, new_user_vector):
    A = np.array(matrix_s).astype(float)
    b = np.array(new_user_vector).astype(float)
    
    A_pseudo_inv = np.linalg.pinv(A)
    # P = A * A_pseudo_inv is the Projection Matrix
    P = A @ A_pseudo_inv
    prediction = P @ b
    return prediction

matrix_s=None
eigenvalues=None
final_mat=None

#define four subspaces
rowspace_basis=[]
colspace_basis=[]
nullspace_basis=[]

while True:
    print("----------WELCOME TO THE MUSIC RECOMMENDATION SYSTEM----------\n")
    print("1. Load user listening data")
    print("2. Show user-song interaction matrix")
    print("3. Find user listening patterns")
    print("4. Find similar song patterns")
    print("5. Detect hidden songs/ redundant songs")
    print("6. Remove redundancy from song data")
    print("7. Generate independent listening features")
    print("8. Find latent taste factors")
    print("9. Enter listening counts for a new user (e.g., 5, 0, 0):")
    print("10. Exit")
    print("\n")

    try:
        choice = int(input("\nEnter choice: "))
    except ValueError:
        continue

    if choice==10:
        print("Exiting.....")
        break

    elif choice!=1 and matrix_s is None:
        print("Please load data first(option 1)\n")
        continue

    if choice==1:
        filename=input("Enter filename:")
        print("Loading Data.....\n")
        print("Data Loaded Successfully!\n")
        rref,pivots,matrix_s=rref_from_csv(filename)
        rowspace_basis,colspace_basis,nullspace_basis = spaces(rref,pivots,matrix_s)
    
    elif choice==2:
        print("-----User-Song Interaction Matrix-----\n")
        print(matrix_s)

    elif choice==3:
        print("-----User Listening Patterns-----\n")
        print(rowspace_basis)
    
    elif choice==4:
        print("-----Similar Song Patterns-----\n")
        print(colspace_basis)
    
    elif choice==5:
        print("-----Hidden Songs/ Redundant Songs-----\n")
        print(nullspace_basis)
    
    elif choice==6:
        print("-----Song Data Without Redundancy-----\n")
        print(rref) 
    
    elif choice==7:
        result=orthogonalize(rowspace_basis)
        print("-----Independent Listening Features-----\n")
        print(result)

    elif choice==8:
       eigenvalues, final_mat = diagonalize(matrix_s)
       print("-----Latent Taste Factors-----\n")
       print(eigenvalues)

    elif choice==9:
        if final_mat is None:
            print("Enter listening counts for a new user (e.g., 5, 0, 0):")
        # Example: if they only listened to Song 1 five times
        user_input = input("Counts: ").split(',')
        new_vector = [float(x) for x in user_input]
        
        prediction = predict_recommendation(matrix_s, new_vector)
        print("\n----- Predicted Listening Profile -----")
        print(prediction)
        print("Note: Higher values for songs they haven't heard are your recommendations!")

    else:
        print("Invalid Choice, Try Again!\n")
        continue

filename = "main_data.csv"
rref,pivots,matrix_s = rref_from_csv(filename)
#rref == rowspace 
rowspace  = rref
#if rref is not None: 
    #print(rref)
    #print(pivots)
rs,cs,ns = spaces(rref,pivots,matrix_s)
Q = orthogonalize(rs) # orthogonalized basis 
#print(Q)
