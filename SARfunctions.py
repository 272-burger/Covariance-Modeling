import numpy as np
import pandas as pd
import libpysal.weights as lpw

# function that makes coordincates of data points
def make_coords(N_points, xm, xM, ym, yM):
    # Define the range of coordinate values
    xmin, ymin = xm, ym
    xmax, ymax = xM, yM
    
    x_coords = np.random.uniform(xmin, xmax, size=N_points)
    y_coords = np.random.uniform(ymin, ymax, size=N_points)
    coords = np.column_stack((x_coords, y_coords))
    return coords

# function that calculates distances between coordinates
def distances(coords):
    N = len(coords)
    
    # Initialize an empty distances matrix
    distances = np.zeros((N, N))

    # Compute the euclidean distance between pair of coords
    for i in range(N):
        for j in range(i+1, N):
            distances[i,j] = distances[j,i] = np.linalg.norm(coords[i] - coords[j])       

    return distances

# function that makes a weight matrix
def make_weight(coords, thres):
    w = lpw.DistanceBand.from_array(coords, threshold=thres, binary=True)
    w.transform = 'R'
    return w.full()[0]


# function that checks if the matrix is invertible
def check_invertible(matrix):
    try:
        inverse = np.linalg.inv(matrix)
        return inverse
    except np.linalg.LinAlgError:
        return False

    
# function that calculates A
def get_A(coords, N, thres, rho):
    
    # make weight matrix
    W = make_weight(coords, thres)
    
    # calculate (I - rho * W)
    I = np.identity(N)
    X = I - rho * W
    
    # get A
    A = check_invertible(X)
    
    return A

# simulation function
def simulation(s, coords, N, threshold, rho): # s = number of simulation
    # calculate invertible matrix A
    A = get_A(coords, N, threshold, rho)
    
    results = [] # generated n * y = [y1, y2, ..., yN]
    V = [] # randomly generated v = [v1, v2, ..., vD], (D = N)

    # run simulations
    for i in range(s):
        # Generate random v from standard normal distribution
        v = np.random.randn(N, 1) 
        V.append(v)
        # Compute the matrix product
        y = np.dot(A, v) # y = N x 1 vector
        results.append(y)
        
        # if i % 10 == 0:
        #     print(f"Iteration: {i+1} / {n}")
        #     print("The shape of the simulation result is: ", np.shape(results))

    return results, V
    # print("The shape of the simulation result is: ", np.shape(results)) # (n, N, 1), cf. 3D array
    

# function that converts results into dataframe (n, N) 
def results_to_df(results):
    s = np.shape(results)[0]
    N = np.shape(results)[1]
    
    # make a column names list; y1, y2, ..
    col_names = []

    for i in range(N):
        col_names.append('y{}'.format(i+1))

    simul_df = pd.DataFrame(np.reshape(results, (s, N)), columns=col_names)

    return simul_df

# function that make distance matrix
# initialize an empty distances matrix
def distance_matrix(coords):
    N = len(coords)
    
    distances = np.zeros((N, N))

    # compute the euclidean distance between pair of coords
    for i in range(N):
        for j in range(i+1, N):
            distances[i,j] = distances[j,i] = np.linalg.norm(coords[i] - coords[j]) 
            
    return distances   

# function that makes a pair dataframe 
def pairs_df(pairs_matrix):
    
    s = np.shape(pairs_matrix)[0]
    N = np.shape(pairs_matrix)[1]
    
    # make a column names list; y1, y2, ..
    col_names = []

    for i in range(N):
        col_names.append('y{}'.format(i+1))
    
    pairs_dataframe = pd.DataFrame(pairs_matrix, columns=col_names, index=col_names)
    return pairs_dataframe