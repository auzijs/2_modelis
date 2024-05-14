import cvxpy as cp
import random
from warehouse_gen import generate_warehouse_dataset, create_distance_matrix, create_warehouse_graph, draw_warehouse_graph
import numpy as np
import time

def extract_route(K, N_0, x):
    routes = {}
    used_locations = []
    for v in range(1, K + 1):
        current_location = 0
        route = [current_location]
        while True:
            next_location = None
            for j in N_0:
                if (current_location, j) not in used_locations and x_values[current_location, j] == 1:
                    next_location = j
                    used_locations.append((current_location, j))
                    break
            if next_location is None or next_location == 0:
                break
            route.append(next_location)
            current_location = next_location
        route.append(0)
        routes[v] = route
    return routes

def print_stats(problem, x_values, u_values, t, nCount, vehicleCount):
    print("Status:", problem.status)
    print("Objective:", problem.value)
    print('\n')
    total_distance = 0
    total_time = 0
    N = [*range(1,nCount+1)]
    N_0 = [0] + N
    used_locations = []
    for i in range(1, vehicleCount + 1):
        vehicle_distance = 0
        vehicle_time = 0
        route = ['Depot']
        current_location = 0

        while True:
            next_location = None
            for j in N_0:
                if (current_location, j) not in used_locations and x_values[current_location, j] == 1:
                    next_location = j
                    used_locations.append((current_location, j))
                    break

            if next_location != None:
                vehicle_distance += t[current_location, next_location]
                vehicle_time += t[current_location, next_location]
            if next_location is None or next_location == 0:
                break
            route.append(str(next_location))
            current_location = next_location

        route.append('Depot')
        print(f"Vehicle {i} route: {' -> '.join(route)}")    
        total_distance += vehicle_distance
        total_time += vehicle_time   

        print(f"Total distance traveled by vehicle {i}: {vehicle_distance}")
        print(f"Total time taken by vehicle {i}: {vehicle_time}")
        print('\n')

    print(f"Total distance traveled by all vehicles: {total_distance}")
    print(f"Total time taken by all vehicles: {total_time}")

seed = 10
random.seed(seed)

C = 90
K = 4
T = 100
nCount = 18

num_cols_before_M = 21
num_cols_after_M = 24
num_cols_total = num_cols_before_M + num_cols_after_M
num_rows = 16
T_z = 1 #first cross aisle col
M_z = (num_cols_before_M + num_cols_after_M) // 2 + 2 #second cross aisle col
B_z = num_cols_total + 3 #third cross aisle col

locations_df = generate_warehouse_dataset(nCount, num_cols_before_M, num_cols_total, num_rows, 15, seed)
locations_df.to_csv('locations.csv', index=False)
distance_matrix_dict = create_distance_matrix(locations_df, T_z, M_z, B_z)
t = np.array([[distance_matrix_dict[(i, j)] for j in range(nCount+1)] for i in range(nCount+1)])

N = [*range(1,nCount+1)]
N_0 = [0] + N


t_param = cp.Parameter((nCount+1, nCount+1), integer=True)
t_param.value = t

d_param = cp.Parameter(nCount+1, nonneg=True)
d_param.value = [0] + [demand for loc, demand in zip(locations_df['Location'], locations_df['Demand']) if loc != 0]

x = cp.Variable((nCount + 1, nCount + 1), boolean=True) 
u = cp.Variable(nCount + 1)
a = cp.Variable(nCount + 1) 

objective = cp.Minimize(cp.sum(cp.multiply(t_param, x)))

constraints = []

for n in N:
    constraints.append(cp.sum([x[i, n] for i in N_0 if i != n]) == 1)
    constraints.append(cp.sum([x[n, j] for j in N_0 if j != n]) == 1)

constraints.append(cp.sum([x[0, j] for j in N]) == K)
constraints.append(cp.sum([x[i, 0] for i in N]) == K)


for i in N:
    for j in N:
        if i != j:
            constraints.append(u[i] + d_param[j] - C * (1 - x[i, j]) <= u[j])

for i in N:
    constraints.append(u[i] >= d_param[i])
    constraints.append(u[i] <= C)


for i in N:
    for j in N:
        if i != j:
            constraints.append(a[i] + t_param[i, j] - T * (1 - x[i, j]) <= a[j])

for i in N:
    constraints.append(t_param[1, i] <= a[i])


problem = cp.Problem(objective, constraints)
start_time = time.time()
problem.solve(solver=cp.CPLEX, verbose=True)
end_time = time.time()

x_values = x.value
u_values = u.value
print_stats(problem, x_values, u_values, t, nCount, K)
print("Solution time: ", end_time - start_time)
routes = extract_route(K, N_0, x_values)
G = create_warehouse_graph(locations_df, num_cols_before_M, num_cols_after_M, num_rows, routes)
draw_warehouse_graph(G, locations_df, True)

