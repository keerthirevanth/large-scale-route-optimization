# Large-Scale Route Optimization on Real Road Networks

Exact and heuristic Travelling Salesman Problem (TSP) solving over **1,000 Indian cities** using
**real OSRM driving distances** — not straight-line approximations. The project scales a textbook
MTZ formulation from N=10 up to N=1,000 and studies what actually makes a large MILP tractable:
warm starts, candidate-arc sparsification, and a specialised exact solver.

## Highlights

- Modelled a 1,000-city delivery tour on real OSRM road distances as a MILP with 999K binary decision variables
- Warm-started Gurobi with Google OR-Tools heuristics, cutting the N=1,000 MIP gap from 71.9% down to 11.0%
- Proved the true global optimum of 48,574 km with Concorde in 26 seconds, closing a gap the MILP could not
- Sparsified via symmetric K-NN at N=500, cutting ~249K possible candidate arcs down to 6,178 at a 3.12% gap
- Generated a 1,000-city heuristic route with OR-Tools in 30s vs. Gurobi's 1,800s time-limited runs

## Repository layout

```
part1_milp_baseline/
  TSP_Gurobi.py                       # MTZ MILP in Gurobi, N = 10 … 1000, 7200s limit + Folium/OSRM maps
  maps/TSP_Map_N*.html                # interactive Folium maps of each solved tour, drawn on real roads
part2_scaling_and_sparsification/
  task1_ortools_heuristic.ipynb       # OR-Tools GUIDED_LOCAL_SEARCH baseline, 30s per instance
  task2_warm_start_gurobi.py          # cold start vs. OR-Tools MIP-start injection into Gurobi
  task3_knn_sparsification.py         # symmetric K-nearest-neighbour arc sparsification at N = 500
reports/
  part1_report.pdf                    # write-up: formulation, results table, maps
  part2_report.pdf                    # write-up: heuristics, warm starts, sparsification, Concorde
  ai_usage_declaration.pdf
```

## The model

A single-vehicle, single-depot TSP on a directed complete graph, solved as a MILP:

- **Decision variables** — `x[i,j] ∈ {0,1}` for every ordered pair `i ≠ j`, i.e. `N(N-1)` binaries
  (999,000 at N=1,000). Self-loops are never created rather than being suppressed with Big-M.
- **Objective** — minimise `Σ d[i,j] · x[i,j]`, where `d` is the OSRM road distance in km.
- **Degree constraints** — every city is left exactly once and entered exactly once.
- **Subtour elimination** — Miller–Tucker–Zemlin, with continuous position variables
  `u[i] ∈ [1, N-1]` and `u[i] - u[j] + N·x[i,j] ≤ N-1`. This keeps the model compact
  (polynomial in N) at the cost of a weak LP relaxation — which is exactly what drives the
  large MIP gaps at N ≥ 500 and motivates Part 2.

## Scaling techniques studied

**1. OR-Tools heuristic (Task 1).** `PATH_CHEAPEST_ARC` construction followed by
`GUIDED_LOCAL_SEARCH`, capped at 30 seconds per instance. It returns a full 1,000-city tour in
half a minute — where the MILP, given 1,800 seconds, still terminates time-limited with a
double-digit gap.

**2. Warm starting (Task 2).** The OR-Tools tour is injected into Gurobi as a MIP start: all
`x[i,j].Start` are zeroed, the arcs on the heuristic tour are set to 1, and the MTZ position
variables `u[i].Start` are set to each city's rank in the sequence — so the incumbent is
feasible for the MTZ constraints on arrival, not just for the degree constraints. Handing
Gurobi a good incumbent immediately collapses the upper bound and drops the N=1,000 gap from
71.9% to 11.0% under the same 1,800s budget.

**3. Symmetric K-NN sparsification (Task 3).** At N=500 the full arc set is ~249K variables.
For each city only its K nearest neighbours are kept as candidate arcs, then the arc set is
**symmetrised** (if `i→j` survives, `j→i` is forced back in) to avoid dead ends, and the depot's
own neighbourhood is explicitly protected against isolation. K=20 keeps just 6,178 arcs —
about 2.5% of the original model — and still lands within 3.12% of optimal. K=10 goes too far:
the candidate graph fragments into disconnected clusters and Gurobi reports the model infeasible.

**4. Concorde as the exact reference.** The MTZ MILP could not close the N=1,000 instance in
any time budget tried. Concorde — a dedicated branch-and-cut TSP solver using subtour and comb
cuts instead of MTZ — **proved** the global optimum of **48,574 km in 26 seconds**. The gap was
never about hardware; it was about the formulation.

## Data

Two CSVs are expected next to the scripts and are **not** included in this repository:

| File | Columns |
|---|---|
| `india_cities_1000.csv` | `Place_Name`, `Latitude`, `Longitude` |
| `distance_matrix.csv` | `fromplace`, `toplace`, `dist_km` (all 1000 × 1000 OSRM pairs) |

Distances come from the OSRM routing engine, so the matrix is **asymmetric** and reflects
actual drivable roads. Map rendering re-queries the public OSRM API
(`router.project-osrm.org`) for tour geometry, batched in chunks of 50 coordinates because the
public endpoint rejects requests with more than 100 waypoints.

## Running

```bash
pip install -r requirements.txt
```

Gurobi needs a licence. The scripts read `GRB_LICENSE_FILE` from the environment; otherwise
`gurobi.lic` in your home directory is used as the fallback.

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic

python part1_milp_baseline/TSP_Gurobi.py                            # N = 10 … 1000, writes maps + results CSV
python part2_scaling_and_sparsification/task2_warm_start_gurobi.py  # cold vs. warm start comparison
python part2_scaling_and_sparsification/task3_knn_sparsification.py # K ∈ {10, 20, 50} at N = 500
```

Each script writes its results table to CSV alongside the console output. The Part 1 script also
saves one interactive `TSP_Map_N<n>.html` per instance — open any file in `part1_milp_baseline/maps/`
in a browser to pan around the optimised tour.

Full runs are long by design: Part 1 allows 7,200s per instance, Part 2 allows 1,800s, and
OR-Tools is capped at 30s.

## Stack

Gurobi (`gurobipy`) · Google OR-Tools · Concorde · OSRM · NumPy · pandas · Folium
