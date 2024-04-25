import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_warehouse_dataset(num_locations, num_cols_before_M, num_cols_total, num_rows, c_limit, seed):
    random.seed(seed)
    data = {'Location': [], 'Col': [], 'Row': [], 'Demand': []}
    
    depot_col = 0
    depot_row = 1
    
    data['Location'].append(0)
    data['Col'].append(depot_col)
    data['Row'].append(depot_row)
    data['Demand'].append(0)

    m_col = num_cols_before_M + 2

    used_positions = []
    
    for i in range(1, num_locations + 1):
        while True:
            col = random.choice([a for a in range(2, num_cols_total + 3) if a != m_col])
            row = random.randint(1, num_rows)
            if (col, row) not in used_positions:
                used_positions.append((col, row))
                demand = random.randint(1, c_limit)
                data['Location'].append(i)
                data['Col'].append(col)
                data['Row'].append(row)
                data['Demand'].append(demand)
                break

    return pd.DataFrame(data)


def calculate_warehouse_distance(x1, z1, x2, z2, B_z, M_z, T_z):
    if x1 == x2:
        return abs(z2 - z1)
    else:
        return min(
            abs(z1 - B_z) + abs(x2 - x1) + abs(z2 - B_z),
            abs(z1 - M_z) + abs(x2 - x1) + abs(z2 - M_z),
            abs(z1 - T_z) + abs(x2 - x1) + abs(z2 - T_z)
        )


def create_distance_matrix(df, B_z, M_z, T_z):
    distance_matrix = {}
    for i in range(df.shape[0]):
        for j in range(df.shape[0]):
            loc1 = df.iloc[i]['Location']
            loc2 = df.iloc[j]['Location']
            col1, row1 = df.iloc[i]['Col'], df.iloc[i]['Row']
            col2, row2 = df.iloc[j]['Col'], df.iloc[j]['Row']
            
            if loc1 == loc2:
                distance_matrix[(loc1, loc2)] = 0
            else:
                distance = calculate_warehouse_distance(row1, col1, row2, col2, T_z, M_z, B_z)
                distance_matrix[(loc1, loc2)] = distance

    return distance_matrix

def create_warehouse_graph(df, num_cols_before_M, num_cols_after_M, num_rows, routes=None):
    G = nx.DiGraph()
    location_to_node = {}


    for row in range(1, num_rows + 1):
        G.add_node(f'FrontRow_{row}', pos=(1, row), type='cross_row')
        G.add_node(f'MiddleRow_{row}', pos=(num_cols_before_M + 2, row), type='cross_row')
        G.add_node(f'BackRow_{row}', pos=(num_cols_before_M + num_cols_after_M + 3, row), type='cross_row')

    for col in range(2, num_cols_before_M + num_cols_after_M + 3):
        if col == num_cols_before_M + 2:
            continue
        for row in range(1, num_rows + 1):
            node_id = f'Col{col}_Row{row}'
            if not df[(df['Col'] == col) & (df['Row'] == row)].empty:
                node_type = 'pickup_node'
                dataset_row = df[(df['Col'] == col) & (df['Row'] == row)].iloc[0]
                location = dataset_row['Location']
                G.add_node(node_id, pos=(col, row), type=node_type, location=location)
                location_to_node[location] = node_id
            else:
                node_type = 'reg_node'
                G.add_node(node_id, pos=(col, row), type=node_type)
            
            if col != 2 and col != num_cols_before_M + 3:
                prev_node_id = f'Col{col-1}_Row{row}'
                G.add_edge(prev_node_id, node_id, arrows=False)

    depot_info = df[df['Location'] == 0].iloc[0]
    depot_x, depot_y = depot_info['Col'], depot_info['Row']
    G.add_node('Depot', pos=(depot_x, depot_y), type='depot')
    location_to_node[0] = 'Depot'

    if routes:
        for vehicle, route in routes.items():
            previous_location_id = route[0]
            previous_node = location_to_node[previous_location_id]
            for location_id in route[1:]:
                current_node = location_to_node[location_id]
                G.add_edge(previous_node, current_node, vehicle=vehicle)
                previous_node = current_node

    return G



def draw_warehouse_graph(G, df, routes=False):
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = {(u, v): data['vehicle'] for u, v, data in G.edges(data=True) if 'vehicle' in data}

    color_map = []
    labels = {}
    for node, data in G.nodes(data=True):
        if data['type'] == 'depot':
            color_map.append('black')
        elif data['type'] == 'pickup_node':
            color_map.append('green')
            location = data['location']
            demand = df[df['Location'] == location]['Demand'].sum()
            labels[node] = demand
        elif data['type'] == 'cross_row':
            color_map.append('red')
        else:
            color_map.append('blue')

    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=50)

    if (routes):
        non_route_edges = [(u, v) for u, v, data in G.edges(data=True) if 'vehicle' not in data]
        nx.draw_networkx_edges(G, pos, edgelist=non_route_edges, edge_color='gray', arrows=False)

        route_edges = [(u, v) for u, v, data in G.edges(data=True) if 'vehicle' in data]
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='orange', arrows=True)
    else:
        non_route_edges = [(u, v) for u, v, data in G.edges(data=True) if 'vehicle' not in data]
        nx.draw_networkx_edges(G, pos, edgelist=non_route_edges, edge_color='gray', arrows=False)

    if (routes):
        route_edges = [(u, v) for u, v, data in G.edges(data=True) if 'vehicle' in data]
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='orange', arrows=True)

    adjusted_edge_labels = {}
    node_radius = 0.1 
    if routes:
        for (u, v), label in edge_labels.items():
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            close_to_node = False
            for node, node_pos in pos.items():
                if np.sqrt((mid_x - node_pos[0])**2 + (mid_y - node_pos[1])**2) < node_radius:
                    close_to_node = True
                    break
            if close_to_node:
                mid_x += 0.5
            mid_y += 0.1
            adjusted_edge_labels[(u, v)] = (label, (mid_x, mid_y))
        for (u, v), (label, label_pos) in adjusted_edge_labels.items():
            plt.text(label_pos[0], label_pos[1], s=label, color='black', horizontalalignment='center')

    for node, demand in labels.items():
        node_pos = pos[node]
        plt.text(node_pos[0], node_pos[1] + 0.1, s=demand, fontsize=10, color='green', verticalalignment='bottom', horizontalalignment='center')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Demand at pickup locations', markersize=10, markerfacecolor='green'),
        plt.Line2D([0], [0], marker='o', color='w', label='Locations', markersize=10, markerfacecolor='blue'),
        plt.Line2D([0], [0], marker='o', color='w', label='Cross aisles', markersize=10, markerfacecolor='red'),
    ]
    if routes:
        legend_elements.append(plt.Line2D([0], [0], linestyle='solid', color='orange', label='Vehicle number and route'))

    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)

    plt.axis('off')
    plt.show()