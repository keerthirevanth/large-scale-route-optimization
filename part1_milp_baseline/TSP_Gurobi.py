import os
# Gurobi licence: set GRB_LICENSE_FILE in your environment, or drop gurobi.lic in $HOME
os.environ.setdefault("GRB_LICENSE_FILE", os.path.expanduser("~/gurobi.lic"))
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import folium
import requests

# --- 1. Load Data ---
CITIES_FILE = "india_cities_1000.csv"
DISTANCES_FILE = "distance_matrix.csv"

cities_df = pd.read_csv(CITIES_FILE)
distances_df = pd.read_csv(DISTANCES_FILE)

# Create a safe mapping dictionary from City Name to its Index
city_list = cities_df['Place_Name'].tolist()
city_to_idx = {name: idx for idx, name in enumerate(city_list)}

# Initialize an empty 1000x1000 matrix
full_dist_matrix = np.zeros((1000, 1000))

# Map the string names in the CSV to their proper integer coordinates
distances_df['from_idx'] = distances_df['fromplace'].map(city_to_idx)
distances_df['to_idx'] = distances_df['toplace'].map(city_to_idx)

# Safely populate the matrix regardless of how the CSV was sorted
full_dist_matrix[distances_df['from_idx'], distances_df['to_idx']] = distances_df['dist_km']

# The exact instance sizes requested in the assignment brief
N_SIZES = [10, 50, 100, 250, 500, 750, 1000]


# --- 2. OSRM Helper Function ---
def get_osrm_route(coords):
    """
    Takes a sequence of coordinates and queries the OSRM routing API to get the 
    actual road geography. We batch in chunks of 50 because public OSRM servers
    will block queries with >100 coordinates.
    """
    route_geometry = []
    batch_size = 50
    
    # Step by batch_size - 1 so the end of one batch connects perfectly to the start of the next
    for i in range(0, len(coords) - 1, batch_size - 1):
        chunk = coords[i : i + batch_size]
        if len(chunk) < 2:
            break
            
        # OSRM expects format: "lon,lat;lon,lat"
        coord_str = ";".join([f"{lon},{lat}" for lat, lon in chunk])
        url = f"http://router.project-osrm.org/route/v1/driving/{coord_str}?overview=full&geometries=geojson"
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "routes" in data and len(data["routes"]) > 0:
                    geom = data["routes"][0]["geometry"]["coordinates"]
                    # Flip from [lon, lat] back to [lat, lon] for Folium mapping
                    route_geometry.extend([[lat, lon] for lon, lat in geom])
        except Exception as e:
            print(f"OSRM Request failed for a chunk: {e}")
            pass
            
    return route_geometry


# --- 3. Main Optimization Function ---
def solve_tsp(n):
    print(f"\n--- Solving for N = {n} ---")
    
    # Slice the data for the current number of cities
    dist_matrix = full_dist_matrix[:n, :n]
    instance_cities = cities_df.head(n)
    
    # Initialize Gurobi environment and model
    env = gp.Env()
    model = gp.Model(f"TSP_India_{n}", env=env)
    
    # Assignment Rule: Hard time limit of 7200 seconds (120 minutes)
    model.setParam('TimeLimit', 7200)
    
    # Decision Variables
    # We only create valid paths where start and end cities are different.
    # This completely solves the "self-loop" problem without needing Big M constraints.
    valid_paths = [(i, j) for i in range(n) for j in range(n) if i != j]
    x = model.addVars(valid_paths, vtype=GRB.BINARY, name="x")
    
    # u[i] = continuous dummy variable for MTZ subtour elimination (for cities 1 to n-1)
    u = model.addVars(range(1, n), lb=1, ub=n-1, vtype=GRB.CONTINUOUS, name="u")
    
    # Objective: Minimize total driving distance in kilometers
    obj = gp.quicksum(dist_matrix[i, j] * x[i, j] for i, j in valid_paths)
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints: Leave every city exactly once
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n) if i != j) == 1 for i in range(n)), name="leave")
    
    # Constraints: Enter every city exactly once
    model.addConstrs((gp.quicksum(x[i, j] for i in range(n) if i != j) == 1 for j in range(n)), name="enter")
    
    # Constraints: MTZ Subtour Elimination
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f"mtz_{i}_{j}")
                
    # Run the solver
    model.optimize()
    
    # Map Gurobi status codes to human-readable text for our table
    status_dict = {GRB.OPTIMAL: "Optimal", GRB.TIME_LIMIT: "Time-limited", GRB.INFEASIBLE: "Infeasible"}
    status = status_dict.get(model.Status, "Unknown")
    
    runtime = model.Runtime
    
    # If the solver found at least one valid route
    if model.SolCount > 0:
        obj_val = model.ObjVal
        obj_bound = model.ObjBound
        mip_gap = model.MIPGap * 100.0  # Convert to percentage
        
        # Build the final sequence of cities starting at the Depot (0)
        tour = [0] 
        curr = 0
        while len(tour) < n:
            for j in range(n):
                if curr != j and x[curr, j].X > 0.5:  # If path is chosen
                    tour.append(j)
                    curr = j
                    break
        tour.append(0)  # Close the loop by returning to the depot
        
        # --- 4. Plotting the HTML Map ---
        print(f"Generating HTML map for N={n}...")
        tour_coords = [(instance_cities.loc[idx, 'Latitude'], instance_cities.loc[idx, 'Longitude']) for idx in tour]
        
        # Initialize map centered at the depot
        m = folium.Map(location=[tour_coords[0][0], tour_coords[0][1]], zoom_start=5)
        
        # Add city markers
        for idx in tour[:-1]: # exclude the final return trip for plotting markers
            is_depot = (idx == 0)
            folium.Marker(
                location=[instance_cities.loc[idx, 'Latitude'], instance_cities.loc[idx, 'Longitude']],
                popup=f"{'DEPOT: ' if is_depot else 'City: '}{instance_cities.loc[idx, 'Place_Name']}",
                icon=folium.Icon(color="red" if is_depot else "blue")
            ).add_to(m)
            
        # Try getting the actual road path from OSRM
        road_path = get_osrm_route(tour_coords)
        if road_path and len(road_path) > 0:
            folium.PolyLine(road_path, color="blue", weight=3).add_to(m)
        else:
            # Fallback to dotted straight lines if OSRM API drops the request
            print("OSRM routing failed or timed out, drawing straight lines instead.")
            folium.PolyLine(tour_coords, color="red", weight=2, dash_array="5,5").add_to(m)
            
        map_filename = f"TSP_Map_N{n}.html"
        m.save(map_filename)
        
        return {
            "N (cities)": n,
            "Optimal / Best Distance (km)": f"{obj_val:.2f} / {obj_bound:.2f}",
            "Runtime": f"{runtime:.2f} sec",
            "MIP Gap (%)": f"{mip_gap:.2f}%",
            "Solver Status": status,
            "Notes": f"Map saved as {map_filename}"
        }
    else:
        # If the model timed out before finding even a single valid route
        return {
            "N (cities)": n,
            "Optimal / Best Distance (km)": "N/A",
            "Runtime": f"{runtime:.2f} sec",
            "MIP Gap (%)": "N/A",
            "Solver Status": status,
            "Notes": "No solution found within time limit."
        }


# --- 5. Execution Block ---
if __name__ == "__main__":
    results = []
    
    # Loop through all required instance sizes
    for n in N_SIZES:
        res = solve_tsp(n)
        results.append(res)
        
    # Print exactly as required for the assignment report
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    final_df = pd.DataFrame(results)
    print(final_df.to_string(index=False))
    
    # Save the table to a CSV so you can easily copy-paste it into your DOCX/PDF
    final_df.to_csv("TSP_Assignment_Results.csv", index=False)