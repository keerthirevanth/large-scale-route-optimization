import os
import time
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Gurobi licence: set GRB_LICENSE_FILE in your environment, or drop gurobi.lic in $HOME
os.environ.setdefault("GRB_LICENSE_FILE", os.path.expanduser("~/gurobi.lic"))

# --- 1. Load Data ---
CITIES_FILE = "india_cities_1000.csv"
DISTANCES_FILE = "distance_matrix.csv"

print("Loading data...")
cities_df = pd.read_csv(CITIES_FILE)
distances_df = pd.read_csv(DISTANCES_FILE)

city_list = cities_df['Place_Name'].tolist()
city_to_idx = {name: idx for idx, name in enumerate(city_list)}
full_dist_matrix = np.zeros((1000, 1000))
distances_df['from_idx'] = distances_df['fromplace'].map(city_to_idx)
distances_df['to_idx'] = distances_df['toplace'].map(city_to_idx)
full_dist_matrix[distances_df['from_idx'], distances_df['to_idx']] = distances_df['dist_km']

# Task 3 is specifically for N = 500
N = 500
dist_matrix = full_dist_matrix[:N, :N]
K_VALUES = [10, 20, 50]

# --- 2. K-NN Sparsification & Gurobi Solver ---
def solve_sparsified_tsp(k):
    print(f"\n--- Running Task 3 for K = {k} ---")
    
    # ==========================================
    # VIVA FOCUS: BUILDING THE SYMMETRIZED K-NN SET
    # ==========================================
    candidate_arcs = set()
    
    # Step 1: Find the top K neighbors for every city
    for i in range(N):
        # argsort() sorts indices from smallest distance to largest
        # We slice [1:k+1] because the closest city (index 0) is always the city itself (distance 0)
        closest_cities = np.argsort(dist_matrix[i, :])[1:k+1]
        for j in closest_cities:
            candidate_arcs.add((i, j))
            
    # Step 2: Symmetrize the graph to prevent dead ends (Infeasibility)
    # If a path i->j exists, we MUST force the path j->i to exist
    symmetric_arcs = set()
    for i, j in candidate_arcs:
        symmetric_arcs.add((i, j))
        symmetric_arcs.add((j, i))
        
    # Step 3: Depot Protection
    # Ensure the depot (City 0) has a guaranteed way in and out to prevent isolation
    # We do this by explicitly making sure the depot's closest neighbors are in the set
    depot_neighbors = np.argsort(dist_matrix[0, :])[1:k+1]
    for j in depot_neighbors:
        symmetric_arcs.add((0, j))
        symmetric_arcs.add((j, 0))
        
    # Convert the set back to a sorted list so Gurobi can process it predictably
    valid_paths = sorted(list(symmetric_arcs))
    # ==========================================

    print(f"Original Variables: {N*(N-1)} | Sparsified Variables: {len(valid_paths)}")

    # Initialize Gurobi Environment
    env = gp.Env()
    env.setParam('OutputFlag', 0)
    model = gp.Model(f"TSP_KNN_{k}", env=env)
    
    # Task 3 Rule: 1800 seconds hard limit
    model.setParam('TimeLimit', 1800)
    
    # Decision Variables - CRITICAL: We ONLY pass `valid_paths`, not the full NxN grid!
    x = model.addVars(valid_paths, vtype=GRB.BINARY, name="x")
    u = model.addVars(range(1, N), lb=1, ub=N-1, vtype=GRB.CONTINUOUS, name="u")
    
    # Objective
    obj = gp.quicksum(dist_matrix[i, j] * x[i, j] for i, j in valid_paths)
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # CRITICAL: We can no longer sum over `range(N)`. We must only sum over paths that actually exist in `valid_paths`!
    
    # 1. Leave Constraint: For city i, sum(x[i, j]) = 1, but only for valid outgoing j's
    for i in range(N):
        valid_outgoing_j = [j for (origin, j) in valid_paths if origin == i]
        if valid_outgoing_j: # Safety check
            model.addConstr(gp.quicksum(x[i, j] for j in valid_outgoing_j) == 1, name=f"leave_{i}")
            
    # 2. Enter Constraint: For city j, sum(x[i, j]) = 1, but only for valid incoming i's
    for j in range(N):
        valid_incoming_i = [i for (i, dest) in valid_paths if dest == j]
        if valid_incoming_i: # Safety check
            model.addConstr(gp.quicksum(x[i, j] for i in valid_incoming_i) == 1, name=f"enter_{j}")

    # 3. MTZ Subtour Elimination (Only applied to active arcs)
    for i, j in valid_paths:
        if i != 0 and j != 0: # MTZ never applies to the depot
            model.addConstr(u[i] - u[j] + N * x[i, j] <= N - 1, name=f"mtz_{i}_{j}")
            
    # Solve
    start_time = time.time()
    model.optimize()
    actual_runtime = time.time() - start_time
    
    # Extract Data for Table 3
    status_map = {GRB.OPTIMAL: "Optimal", GRB.TIME_LIMIT: "Time-limited", GRB.INFEASIBLE: "Infeasible"}
    status = status_map.get(model.Status, f"Unknown ({model.Status})")
    
    # Handle the expected "Infeasible" scenario for K=10
    if model.Status == GRB.INFEASIBLE:
        return {
            "K": k,
            "Candidate Arcs (≈)": len(valid_paths),
            "Total Distance (km)": "N/A",
            "Runtime (s)": f"{actual_runtime:.2f}",
            "MIP Gap (%)": "N/A",
            "Status": "Infeasible",
            "Notes": "K too small; map disconnected into isolated clusters."
        }
    
    # Handle standard results
    if model.SolCount > 0:
        gap = model.MIPGap * 100.0 if model.MIPGap < float('inf') else float('inf')
        notes = "Successfully proved optimal" if status == "Optimal" else "Heuristic route found before timeout"
        
        return {
            "K": k,
            "Candidate Arcs (≈)": len(valid_paths),
            "Total Distance (km)": f"{model.ObjVal:.2f}",
            "Runtime (s)": f"{actual_runtime:.2f}",
            "MIP Gap (%)": f"{gap:.2f}%",
            "Status": status,
            "Notes": notes
        }
    else:
        return {
            "K": k,
            "Candidate Arcs (≈)": len(valid_paths),
            "Total Distance (km)": "N/A",
            "Runtime (s)": f"{actual_runtime:.2f}",
            "MIP Gap (%)": "N/A",
            "Status": status,
            "Notes": "Failed to find any feasible integer route within time limit."
        }


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    results = []
    
    print("="*60)
    print("STARTING TASK 3: K-NN SPARSIFICATION (N = 500)")
    print("="*60)
    
    for k in K_VALUES:
        res = solve_sparsified_tsp(k)
        results.append(res)
        
    print("\n" + "="*110)
    print("TABLE 3 - K-NEAREST-NEIGHBOUR SPARSIFICATION, N = 500")
    print("="*110)
    final_df = pd.DataFrame(results)
    print(final_df.to_string(index=False))
    
    final_df.to_csv("TSP_Task3_Results.csv", index=False)