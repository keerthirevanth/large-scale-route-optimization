#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Gurobi licence: set GRB_LICENSE_FILE in your environment, or drop gurobi.lic in $HOME
os.environ.setdefault("GRB_LICENSE_FILE", os.path.expanduser("~/gurobi.lic"))

# --- 1. Load Data ---
CITIES_FILE = "india_cities_1000.csv"
DISTANCES_FILE = "distance_matrix.csv"

print("Loading data and building distance matrix...")
cities_df = pd.read_csv(CITIES_FILE)
distances_df = pd.read_csv(DISTANCES_FILE)

city_list = cities_df['Place_Name'].tolist()
city_to_idx = {name: idx for idx, name in enumerate(city_list)}
full_dist_matrix = np.zeros((1000, 1000))
distances_df['from_idx'] = distances_df['fromplace'].map(city_to_idx)
distances_df['to_idx'] = distances_df['toplace'].map(city_to_idx)
full_dist_matrix[distances_df['from_idx'], distances_df['to_idx']] = distances_df['dist_km']

N_SIZES = [10, 50, 100, 250, 500, 1000]

# --- 2. OR-Tools Helper: Get the Heuristic Tour ---
def get_ortools_tour(n):
    """Runs OR-Tools for 30s and returns the exact sequence of cities."""
    dist_matrix = full_dist_matrix[:n, :n]
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_idx, to_idx):
        return int(dist_matrix[manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)] * 1000)
        
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30 
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        # Extract the sequence of cities
        index = routing.Start(0)
        tour = []
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        tour.append(manager.IndexToNode(index)) # Append the final return to depot
        return tour
    return None

# --- 3. Gurobi MTZ Solver (Handles both Cold and Warm Starts) ---
def solve_gurobi_mtz(n, warm_start_tour=None):
    mode = "WARM" if warm_start_tour else "COLD"
    print(f"\n--- Running Gurobi {mode} Start for N = {n} ---")
    
    dist_matrix = full_dist_matrix[:n, :n]
    env = gp.Env()
    
    # Disable console output to keep your terminal clean during long loops
    env.setParam('OutputFlag', 0) 
    model = gp.Model(f"TSP_{mode}_{n}", env=env)
    
    # Task 2 Rule: 1800 seconds (30 minutes) hard limit
    model.setParam('TimeLimit', 1800)
    
    # Decision Variables
    valid_paths = [(i, j) for i in range(n) for j in range(n) if i != j]
    x = model.addVars(valid_paths, vtype=GRB.BINARY, name="x")
    u = model.addVars(range(1, n), lb=1, ub=n-1, vtype=GRB.CONTINUOUS, name="u")
    
    # Objective
    obj = gp.quicksum(dist_matrix[i, j] * x[i, j] for i, j in valid_paths)
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n) if i != j) == 1 for i in range(n)), name="leave")
    model.addConstrs((gp.quicksum(x[i, j] for i in range(n) if i != j) == 1 for j in range(n)), name="enter")
    
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f"mtz_{i}_{j}")
    
    # ==========================================
    # VIVA FOCUS: THE MIP START INJECTION
    # ==========================================
    if warm_start_tour:
        print("Injecting OR-Tools incumbent into Gurobi...")
        
        # 1. Start by telling Gurobi NOT to take any roads
        for i, j in valid_paths:
            x[i, j].Start = 0
            
        # 2. Iterate through the OR-Tools sequence and turn ON those specific roads
        for k in range(len(warm_start_tour) - 1):
            i = warm_start_tour[k]
            j = warm_start_tour[k+1]
            x[i, j].Start = 1
            
        # 3. Inject the sequence position into the MTZ 'u' variable
        # (e.g., if City 5 is the 3rd stop, set u[5] = 3)
        for k in range(1, len(warm_start_tour) - 1):
            city = warm_start_tour[k]
            u[city].Start = k
    # ==========================================
                
    start_time = time.time()
    model.optimize()
    actual_runtime = time.time() - start_time
    
    # Extract results safely
    if model.SolCount > 0:
        return {
            "Runtime": actual_runtime,
            "ObjVal": model.ObjVal,
            "Gap": model.MIPGap * 100.0 if model.MIPGap < float('inf') else 100.0,
            "Status": model.Status
        }
    else:
        return {
            "Runtime": actual_runtime,
            "ObjVal": float('inf'),
            "Gap": float('inf'),
            "Status": model.Status
        }

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    results = []
    
    for n in N_SIZES:
        print("="*50)
        print(f"EVALUATING N = {n}")
        print("="*50)
        
        # 1. Run Cold
        cold_res = solve_gurobi_mtz(n, warm_start_tour=None)
        
        # 2. Generate Heuristic Tour
        print(f"\nGenerating 30s OR-Tools Heuristic for N = {n}...")
        ortools_tour = get_ortools_tour(n)
        
        # 3. Run Warm
        warm_res = solve_gurobi_mtz(n, warm_start_tour=ortools_tour)
        
        # 4. Process Notes & Best Distance
        best_dist = min(cold_res["ObjVal"], warm_res["ObjVal"])
        notes = []
        if cold_res["Status"] == GRB.TIME_LIMIT: notes.append("Cold: Time-limited")
        if warm_res["Status"] == GRB.TIME_LIMIT: notes.append("Warm: Time-limited")
        notes_str = " | ".join(notes) if notes else "Both proved optimal"
        
        # Format row for Table 2
        results.append({
            "N (cities)": n,
            "Cold-start Runtime (s)": f"{cold_res['Runtime']:.2f}",
            "Cold-start Gap (%)": f"{cold_res['Gap']:.2f}%" if cold_res["Gap"] != float('inf') else "N/A",
            "Warm-start Runtime (s)": f"{warm_res['Runtime']:.2f}",
            "Warm-start Gap (%)": f"{warm_res['Gap']:.2f}%" if warm_res["Gap"] != float('inf') else "N/A",
            "Best Distance (km)": f"{best_dist:.2f}" if best_dist != float('inf') else "N/A",
            "Notes": notes_str
        })
        
    print("\n" + "="*110)
    print("TABLE 2 - GUROBI MTZ: COLD START VS WARM START")
    print("="*110)
    final_df = pd.DataFrame(results)
    print(final_df.to_string(index=False))
    
    final_df.to_csv("TSP_Task2_Results.csv", index=False)

