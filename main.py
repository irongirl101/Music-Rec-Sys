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


def diagonalize(A):
    
    """
    (P,D)=matrix_s.diagonalize
    return P, D
    """

    L=sp.symbols('lam') #assume L is lambda
    I=sp.eye(A.shape[0])

    c_matrix=A-L*I
    c_poly=c_matrix.det()
    eigenvalues=sp.solve(c_poly,L)
    eigenvectors={}
    for eigenval in eigenvalues:
        cval_matrix=A-L*I
        eigenvectors[eigenval]=cval_matrix.nullspace()
    
    vectors=eigenvectors.values()
    P=np.stack(vectors, axis=1)

    #A = P D P⁻¹
    S=np.linalg.inv(P)
    final_mat=S @ A @ P

    return eigenvalues, final_mat

choice=0
#define values for the matrix, its rref and pivots
rref=[[]]
pivots=0
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
    print("9. Transfrom data into latent feature space")
    print("10. Exit")
    print("\n")
    choice=int(input("Enter your choice:"))

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
            print("Please compute latent factors first(Option 8)\n")
            continue
        print("-----Latent Feature Representation-----\n")
        print(final_mat)

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
print(Q)
