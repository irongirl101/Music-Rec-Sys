#import all required libraries
import pandas as pd 
import sympy as sp
import numpy as np 
from scipy.linalg import qr 

#print welcome symbol
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

# STEPS 1 AND 2 - MATRIX AND RREF

#access csv file and make rref from csv file
def rref_from_csv(filename):
    try:
        # load csv
        df = pd.read_csv(filename,header = None)
        
        # convert data to float datatype
        matrix = df.to_numpy(dtype=float)
        # creating the matrix
        matrix_s = sp.Matrix(matrix)
        #making values into in rref format
        rref, pivots = matrix_s.rref()
        #return values in specified format
    
        return rref, pivots, matrix_s
        
    except Exception as e: 
        print(f"Error Occurred: {e}")
        return None, None, None
    
# STEPS 3 AND 4 - SUBSPACES AND BASIS - identifying independent patterns and redundancy 

#Creating row, column and null spaces
def spaces(rref, pivots, matrix_s): 
    #identifying basis vectors in row space
    rowspace_basis = []
    for i in range(rref.rows): 
        row = rref.row(i)
        if not row.is_zero_matrix: 
            rowspace_basis.append(list(row))
    #identifying basis vectors in columnspace
    colspace_basis = [] 
    for j in pivots: 
        col = matrix_s.col(j)
        colspace_basis.append(list(col))
    #finding nullspace 
    nullspace = matrix_s.nullspace()
    nullspace_basis = [list(i) for i in nullspace]
    #return all the basis vectors
    return rowspace_basis, colspace_basis, nullspace_basis

# STEP 5 - Feature Decorrelation - finding independent vectors  

#Orthogonolization of vectors    
def orthogonalize(vectors): 
    # Converts basis vectors into "Pure Genre Axis" (Orthonormal)
    A = np.stack(vectors, axis=1)
    Q, R = qr(A, mode='economic')
    return Q    

# STEP 8 and 9 - EIGEN VALUES AND DIAGONALIZATION - COVARIANCE MATRIX - finds trends and simplifies the data.  

def diagonalize_trends(matrix_s):
    # Calculate the Covariance Matrix (Song-to-Song correlations)
    A = matrix_s
    cov = A.T * A 
    # P = Eigenvector Matrix (Trends), D = Diagonal Matrix (Strength of Trends)
    P, D = cov.diagonalize()
    return P, D

# STEP 6 AND 7 - PROJECTION AND LEAST SQUARES - Best approximate solution for missing data 

def predict_recommendation(matrix_s, new_user_vector):
    A = np.array(matrix_s).astype(float)
    b = np.array(new_user_vector).astype(float)

    if A.shape[1] != len(b):
        print(f"The matrix expects {A.shape[1]} songs, but you entered {len(b)} values.")
        return None

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return A @ x

# CLI 

matrix_s = None
while True:
    print("1. Load User Listening Data")
    print("2. Show User-Song Interaction Matrix")
    print("3. Find User Listening Patterns (Row Space)")
    print("4. Find Similar Song Patterns (Column Space)")
    print("5. Detect Redundancy in Data (Null Space)")
    print("6. Show Cleaned Data (RREF)")
    print("7. Extract Independent Genre Axes (Orthogonalization)")
    print("8. Discover Global Trends (Eigen-Decomposition)")
    print("9. Predict Ratings for a New User (Projection/Least Squares)")
    print("10. Exit")
    
    try:
        choice = int(input("\nEnter choice: "))
    except ValueError:
        continue

    if choice == 10: break
    if choice != 1 and matrix_s is None:
        print("Please load data first.")
        continue

    if choice == 1:
        filename = input("Enter CSV filename: ")
        rref, pivots, matrix_s = rref_from_csv(filename)
        rs, cs, ns = spaces(rref, pivots, matrix_s)
        print("Data Loaded Successfully!")

    elif choice == 2:
        sp.pprint(matrix_s)

    elif choice == 3:
        print(f"Row Space Basis (User Archetypes): {rs}")
    
    elif choice == 4:
        print(f"Column Space Basis (Song Features): {cs}")
    
    elif choice == 5:
        print(f"Null Space (Gaps in listening trends): {ns}")
    
    elif choice == 6:
        sp.pprint(rref)
    
    elif choice == 7:
        result = orthogonalize(rs)
        print("Orthonormal Genre Basis Matrix (Q):")
        print(result)

    elif choice == 8:
        P, D = diagonalize_trends(matrix_s)
        print("Eigenvalues (Diagonal D - Strength of Trends):")
        sp.pprint(D)
        print("\nEigenvectors (Matrix P - Directions of Trends):")
        sp.pprint(P)

    elif choice == 9:
        user_input = input("Enter play counts for Songs 1, 2, 3 (e.g. 5, 0, 0): ")
        new_vector = [float(x) for x in user_input.split(',')]
        prediction = predict_recommendation(matrix_s, new_vector)
        print("\nPredicted Scores (Higher = Stronger Recommendation):")
        print(prediction)

    else: print("Invalid Choice.")

