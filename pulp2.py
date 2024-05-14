import pulp
import random
from warehouse_gen import generate_warehouse_dataset, create_distance_matrix, create_warehouse_graph, draw_warehouse_graph
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
                if (current_location, j) not in used_locations and x[current_location][j].varValue == 1:
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

def print_stats(model, vehicleCount, N, N_0, x, t, u):
    print("Status:", pulp.LpStatus[model.status])
    print('Objective:', pulp.value(model.objective))
    print('\n')
    
    total_distance = 0
    total_time = 0
    used_locations = []

    for i in range(1, vehicleCount + 1):
        vehicle_distance = 0
        vehicle_time = 0
        route = ['Depot']
        current_location = 0
        while True:
            next_location = None
            for j in N_0:
                if (current_location, j) not in used_locations and x[current_location][j].varValue == 1:
                    next_location = j
                    used_locations.append((current_location, j))
                    break

            if (next_location != None):
                vehicle_distance += t[(current_location, next_location)]
                vehicle_time += t[(current_location, next_location)]
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

C = 50
K = 1
T = 100
nCount = 7

num_cols_before_M = 11
num_cols_after_M = 13
num_cols_total = num_cols_before_M + num_cols_after_M
num_rows = 15
T_z = 1 #first cross aisle col
M_z = (num_cols_before_M + num_cols_after_M) // 2 + 2 #second cross aisle col
B_z = num_cols_total + 3 #third cross aisle col

locations_df = generate_warehouse_dataset(nCount, num_cols_before_M, num_cols_total, num_rows, 15, seed)
locations_df.to_csv('locations.csv', index=False)
t = create_distance_matrix(locations_df, T_z, M_z, B_z)

d = {loc: demand for loc, demand in zip(locations_df['Location'], locations_df['Demand']) if loc != 0}

N = [*range(1, nCount + 1)]
N_0 = [0] + N

model = pulp.LpProblem("VehicleRoutingProblem", pulp.LpMinimize)

x = pulp.LpVariable.dicts("x", (N_0, N_0), cat=pulp.LpBinary)
u = pulp.LpVariable.dicts("u", N)
a = pulp.LpVariable.dicts("a", N)

for i in N_0:
    model += x[i][i] == 0

model += pulp.lpSum(t[(i, j)] * x[i][j] for i in N_0 for j in N_0)

for n in N:
    model += pulp.lpSum(x[i][n] for i in N_0 if i != n) == 1
    model += pulp.lpSum(x[n][j] for j in N_0 if j != n) == 1


model += pulp.lpSum(x[0][j] for j in N) == K
model += pulp.lpSum(x[i][0] for i in N) == K

for i in N:
    for j in N:
        if i != j:
            model += u[i] + d[j] - C * (1 - x[i][j]) <= u[j]
            
for i in N:
    model += u[i] >= d[i]
    model += u[i] <= C

for i in N:
    for j in N:
        if i != j:
            model += a[i] + t[(i, j)] - T * (1 - x[i][j]) <= a[j]


for i in N:
    model += pulp.lpSum(t[(1, i)]) <= a[i]

solver = pulp.CPLEX_CMD(msg=1)
start_time = time.time()
model.solve(solver)
end_time = time.time()
print_stats(model, K, N, N_0, x, t, u)
print("Solution time: ", end_time - start_time)
routes = extract_route(K, N_0, x)
G = create_warehouse_graph(locations_df, num_cols_before_M, num_cols_after_M, num_rows, routes)
draw_warehouse_graph(G, locations_df, True)
