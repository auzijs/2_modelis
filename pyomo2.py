from pyomo.environ import *
from pyomo.opt import SolverFactory
import random
from warehouse_gen import generate_warehouse_dataset, create_distance_matrix, create_warehouse_graph, draw_warehouse_graph
import time

def extract_route(model):
    routes = {}
    used_locations = []
    for v in range(1, model.K + 1):
        current_location = 0
        route = [current_location]
        while True:
            next_location = None
            for j in model.N_0:
                if (current_location, j) not in used_locations and model.x[current_location, j].value == 1:
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

def print_stats(model):
    print('Objective:', model.obj())

    print('\n')
    total_distance = 0
    total_time = 0
    used_locations = []

    for v in range(1, model.K + 1):
        vehicle_distance = 0
        vehicle_time = 0
        route = ['Depot']
        current_location = 0
        while True:
            next_location = None
            for j in model.N_0:
                if (current_location, j) not in used_locations and model.x[current_location, j].value == 1:
                    next_location = j
                    used_locations.append((current_location, j))
                    break

            if (next_location != None):
                vehicle_distance += model.t[current_location, next_location]
                vehicle_time += model.t[current_location, next_location]
                if next_location is None or next_location == 0:
                    break
                route.append(str(next_location))
                current_location = next_location

        route.append('Depot')
        print(f"Vehicle {v} route: {' -> '.join(route)}")

        total_distance += vehicle_distance
        total_time += vehicle_time

        print(f"Total distance traveled by vehicle {v}: {vehicle_distance}")
        print(f"Total time taken by vehicle {v}: {vehicle_time}")
        print('\n')


    print(f"Total distance traveled by all vehicles: {total_distance}")
    print(f"Total time taken by all vehicles: {total_time}")


seed = 10
random.seed(seed)

C = 100
K = 1
T = 900
nCount = 44

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
N = [*range(1,nCount+1)] 
N_0 = [0] + N 

model = ConcreteModel()
model.N = Set(initialize=N)   

model.N_0 = Set(initialize=N_0) 

model.t = Param(model.N_0, model.N_0, initialize=t) 
model.d = Param(model.N, initialize=d)
model.K = K
model.C = C
model.T = T

model.u = Var(model.N)
model.a = Var(model.N)

model.x = Var(model.N_0, model.N_0, within=Binary, initialize=0)


def obj_rule(model):
    return sum(model.t[i, j] * model.x[i, j] for i in model.N_0 for j in model.N_0)


model.obj = Objective(rule=obj_rule, sense=minimize)


def visiting_rule1(model, n):
    return sum(model.x[i, n] for i in model.N_0) == 1


model.visiting1 = Constraint(model.N, rule=visiting_rule1)


def visiting_rule2(model, n):
    return sum(model.x[n, j] for j in model.N_0) == 1


model.visiting2 = Constraint(model.N, rule=visiting_rule2)


def num_vehicle_rule1(model):
    return sum(model.x[i, 0] for i in model.N) == model.K


model.num_vehicle1 = Constraint(rule=num_vehicle_rule1)


def num_vehicle_rule2(model):
    return sum(model.x[0, j] for j in model.N) == model.K


model.num_vehicle2 = Constraint(rule=num_vehicle_rule2)


def sub_tour_rule(model, i, j):
    return model.u[i] + model.d[j] - model.C * (1 - model.x[i, j]) <= model.u[j]


model.sub_tour = Constraint(model.N, model.N, rule=sub_tour_rule)


def capacity_rule(model, i):
    return (model.d[i], model.u[i], model.C)


model.capacity = Constraint(model.N, rule=capacity_rule)


def arrival_time_rule(model, i, j):
    return model.a[i] + model.t[i, j] - model.T * (1 - model.x[i, j]) <= model.a[j]


model.arrival_time = Constraint(model.N, model.N, rule=arrival_time_rule)


def time_rule(model, i):
    return model.t[1, i] <= model.a[i]


model.time = Constraint(model.N, rule=time_rule)


opt = SolverFactory('cplex')
start_time = time.time()
opt.solve(model, tee=True)
end_time = time.time()
print_stats(model)
print("Solution time: ", end_time - start_time)
routes = extract_route(model)
G = create_warehouse_graph(locations_df, num_cols_before_M, num_cols_after_M, num_rows, routes)
draw_warehouse_graph(G, locations_df, True)