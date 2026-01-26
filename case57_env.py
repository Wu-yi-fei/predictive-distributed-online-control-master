"""
    Build a GridEnv instance using IEEE 57-bus data from `nodes.case57`
    with a linearized DistFlow-style radial approximation.

    Power flow data for IEEE 57 bus test case.
    Please see L{caseformat} for details on the case file format.

    This data was converted from IEEE Common Data Format
    (ieee57cdf.txt) on 20-Sep-2004 by cdf2matp, rev. 1.11

    Converted from IEEE CDF file from:
    U{http://www.ee.washington.edu/research/pstca/}

    Manually modified C{Qmax}, C{Qmin} on generator 1 to 200, -140,
    respectively.

    08/25/93 UW ARCHIVE           100.0  1961 W IEEE 57 Bus Test Case

    @return: Power flow data for IEEE 57 bus test case.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

from script_2 import GridEnv  # reuse existing environment class

# --- Inlined minimal IEEE 57-bus data (baseMVA, bus, branch) ---
# Source: nodes.case57 (previously imported); kept essential fields only.
baseMVA = 100.0
bus = np.array([
    [1, 3, 55, 17, 0, 0, 1, 1.04, 0, 0, 1, 1.06, 0.94],
    [2, 2, 3, 88, 0, 0, 1, 1.01, -1.18, 0, 1, 1.06, 0.94],
    [3, 2, 41, 21, 0, 0, 1, 0.985, -5.97, 0, 1, 1.06, 0.94],
    [4, 1, 0, 0, 0, 0, 1, 0.981, -7.32, 0, 1, 1.06, 0.94],
    [5, 1, 13, 4, 0, 0, 1, 0.976, -8.52, 0, 1, 1.06, 0.94],
    [6, 2, 75, 2, 0, 0, 1, 0.98, -8.65, 0, 1, 1.06, 0.94],
    [7, 1, 0, 0, 0, 0, 1, 0.984, -7.58, 0, 1, 1.06, 0.94],
    [8, 2, 150, 22, 0, 0, 1, 1.005, -4.45, 0, 1, 1.06, 0.94],
    [9, 2, 121, 26, 0, 0, 1, 0.98, -9.56, 0, 1, 1.06, 0.94],
    [10, 1, 5, 2, 0, 0, 1, 0.986, -11.43, 0, 1, 1.06, 0.94],
    [11, 1, 0, 0, 0, 0, 1, 0.974, -10.17, 0, 1, 1.06, 0.94],
    [12, 2, 377, 24, 0, 0, 1, 1.015, -10.46, 0, 1, 1.06, 0.94],
    [13, 1, 18, 2.3, 0, 0, 1, 0.979, -9.79, 0, 1, 1.06, 0.94],
    [14, 1, 10.5, 5.3, 0, 0, 1, 0.97, -9.33, 0, 1, 1.06, 0.94],
    [15, 1, 22, 5, 0, 0, 1, 0.988, -7.18, 0, 1, 1.06, 0.94],
    [16, 1, 43, 3, 0, 0, 1, 1.013, -8.85, 0, 1, 1.06, 0.94],
    [17, 1, 42, 8, 0, 0, 1, 1.017, -5.39, 0, 1, 1.06, 0.94],
    [18, 1, 27.2, 9.8, 0, 10, 1, 1.001, -11.71, 0, 1, 1.06, 0.94],
    [19, 1, 3.3, 0.6, 0, 0, 1, 0.97, -13.2, 0, 1, 1.06, 0.94],
    [20, 1, 2.3, 1, 0, 0, 1, 0.964, -13.41, 0, 1, 1.06, 0.94],
    [21, 1, 0, 0, 0, 0, 1, 1.008, -12.89, 0, 1, 1.06, 0.94],
    [22, 1, 0, 0, 0, 0, 1, 1.01, -12.84, 0, 1, 1.06, 0.94],
    [23, 1, 6.3, 2.1, 0, 0, 1, 1.008, -12.91, 0, 1, 1.06, 0.94],
    [24, 1, 0, 0, 0, 0, 1, 0.999, -13.25, 0, 1, 1.06, 0.94],
    [25, 1, 6.3, 3.2, 0, 5.9, 1, 0.982, -18.13, 0, 1, 1.06, 0.94],
    [26, 1, 0, 0, 0, 0, 1, 0.959, -12.95, 0, 1, 1.06, 0.94],
    [27, 1, 9.3, 0.5, 0, 0, 1, 0.982, -11.48, 0, 1, 1.06, 0.94],
    [28, 1, 4.6, 2.3, 0, 0, 1, 0.997, -10.45, 0, 1, 1.06, 0.94],
    [29, 1, 17, 2.6, 0, 0, 1, 1.01, -9.75, 0, 1, 1.06, 0.94],
    [30, 1, 3.6, 1.8, 0, 0, 1, 0.962, -18.68, 0, 1, 1.06, 0.94],
    [31, 1, 5.8, 2.9, 0, 0, 1, 0.936, -19.34, 0, 1, 1.06, 0.94],
    [32, 1, 1.6, 0.8, 0, 0, 1, 0.949, -18.46, 0, 1, 1.06, 0.94],
    [33, 1, 3.8, 1.9, 0, 0, 1, 0.947, -18.5, 0, 1, 1.06, 0.94],
    [34, 1, 0, 0, 0, 0, 1, 0.959, -14.1, 0, 1, 1.06, 0.94],
    [35, 1, 6, 3, 0, 0, 1, 0.966, -13.86, 0, 1, 1.06, 0.94],
    [36, 1, 0, 0, 0, 0, 1, 0.976, -13.59, 0, 1, 1.06, 0.94],
    [37, 1, 0, 0, 0, 0, 1, 0.985, -13.41, 0, 1, 1.06, 0.94],
    [38, 1, 14, 7, 0, 0, 1, 1.013, -12.71, 0, 1, 1.06, 0.94],
    [39, 1, 0, 0, 0, 0, 1, 0.983, -13.46, 0, 1, 1.06, 0.94],
    [40, 1, 0, 0, 0, 0, 1, 0.973, -13.62, 0, 1, 1.06, 0.94],
    [41, 1, 6.3, 3, 0, 0, 1, 0.996, -14.05, 0, 1, 1.06, 0.94],
    [42, 1, 7.1, 4.4, 0, 0, 1, 0.966, -15.5, 0, 1, 1.06, 0.94],
    [43, 1, 2, 1, 0, 0, 1, 1.01, -11.33, 0, 1, 1.06, 0.94],
    [44, 1, 12, 1.8, 0, 0, 1, 1.017, -11.86, 0, 1, 1.06, 0.94],
    [45, 1, 0, 0, 0, 0, 1, 1.036, -9.25, 0, 1, 1.06, 0.94],
    [46, 1, 0, 0, 0, 0, 1, 1.05, -11.89, 0, 1, 1.06, 0.94],
    [47, 1, 29.7, 11.6, 0, 0, 1, 1.033, -12.49, 0, 1, 1.06, 0.94],
    [48, 1, 0, 0, 0, 0, 1, 1.027, -12.59, 0, 1, 1.06, 0.94],
    [49, 1, 18, 8.5, 0, 0, 1, 1.036, -12.92, 0, 1, 1.06, 0.94],
    [50, 1, 21, 10.5, 0, 0, 1, 1.023, -13.39, 0, 1, 1.06, 0.94],
    [51, 1, 18, 5.3, 0, 0, 1, 1.052, -12.52, 0, 1, 1.06, 0.94],
    [52, 1, 4.9, 2.2, 0, 0, 1, 0.98, -11.47, 0, 1, 1.06, 0.94],
    [53, 1, 20, 10, 0, 6.3, 1, 0.971, -12.23, 0, 1, 1.06, 0.94],
    [54, 1, 4.1, 1.4, 0, 0, 1, 0.996, -11.69, 0, 1, 1.06, 0.94],
    [55, 1, 6.8, 3.4, 0, 0, 1, 1.031, -10.78, 0, 1, 1.06, 0.94],
    [56, 1, 7.6, 2.2, 0, 0, 1, 0.968, -16.04, 0, 1, 1.06, 0.94],
    [57, 1, 6.7, 2, 0, 0, 1, 0.965, -16.56, 0, 1, 1.06, 0.94],
])

branch = np.array([
    [1, 2, 0.0083, 0.028, 0.129],
    [2, 3, 0.0298, 0.085, 0.0818],
    [3, 4, 0.0112, 0.0366, 0.038],
    [4, 5, 0.0625, 0.132, 0.0258],
    [4, 6, 0.043, 0.148, 0.0348],
    [6, 7, 0.02, 0.102, 0.0276],
    [6, 8, 0.0339, 0.173, 0.047],
    [8, 9, 0.0099, 0.0505, 0.0548],
    [9, 10, 0.0369, 0.1679, 0.044],
    [9, 11, 0.0258, 0.0848, 0.0218],
    [9, 12, 0.0648, 0.295, 0.0772],
    [9, 13, 0.0481, 0.158, 0.0406],
    [13, 14, 0.0132, 0.0434, 0.011],
    [13, 15, 0.0269, 0.0869, 0.023],
    [1, 15, 0.0178, 0.091, 0.0988],
    [1, 16, 0.0454, 0.206, 0.0546],
    [1, 17, 0.0238, 0.108, 0.0286],
    [3, 15, 0.0162, 0.053, 0.0544],
    [4, 18, 0, 0.555, 0],
    [4, 18, 0, 0.43, 0],
    [5, 6, 0.0302, 0.0641, 0.0124],
    [7, 8, 0.0139, 0.0712, 0.0194],
    [10, 12, 0.0277, 0.1262, 0.0328],
    [11, 13, 0.0223, 0.0732, 0.0188],
    [12, 13, 0.0178, 0.058, 0.0604],
    [12, 16, 0.018, 0.0813, 0.0216],
    [12, 17, 0.0397, 0.179, 0.0476],
    [14, 15, 0.0171, 0.0547, 0.0148],
    [18, 19, 0.461, 0.685, 0],
    [19, 20, 0.283, 0.434, 0],
    [21, 20, 0, 0.7767, 0],
    [21, 22, 0.0736, 0.117, 0],
    [22, 23, 0.0099, 0.0152, 0],
    [23, 24, 0.166, 0.256, 0.0084],
    [24, 25, 0, 1.182, 0],
    [24, 25, 0, 1.23, 0],
    [24, 26, 0, 0.0473, 0],
    [26, 27, 0.165, 0.254, 0],
    [27, 28, 0.0618, 0.0954, 0],
    [28, 29, 0.0418, 0.0587, 0],
    [7, 29, 0, 0.0648, 0],
    [25, 30, 0.135, 0.202, 0],
    [30, 31, 0.326, 0.497, 0],
    [31, 32, 0.507, 0.755, 0],
    [32, 33, 0.0392, 0.036, 0],
    [34, 32, 0, 0.953, 0],
    [34, 35, 0.052, 0.078, 0.0032],
    [35, 36, 0.043, 0.0537, 0.0016],
    [36, 37, 0.029, 0.0366, 0],
    [37, 38, 0.0651, 0.1009, 0.002],
    [37, 39, 0.0239, 0.0379, 0],
    [36, 40, 0.03, 0.0466, 0],
    [22, 38, 0.0192, 0.0295, 0],
    [11, 41, 0, 0.749, 0],
    [41, 42, 0.207, 0.352, 0],
    [41, 43, 0, 0.412, 0],
    [38, 44, 0.0289, 0.0585, 0.002],
    [15, 45, 0, 0.1042, 0],
    [14, 46, 0, 0.0735, 0],
    [46, 47, 0.023, 0.068, 0.0032],
    [47, 48, 0.0182, 0.0233, 0],
    [48, 49, 0.0834, 0.129, 0.0048],
    [49, 50, 0.0801, 0.128, 0],
    [50, 51, 0.1386, 0.22, 0],
    [10, 51, 0, 0.0712, 0],
    [13, 49, 0, 0.191, 0],
    [29, 52, 0.1442, 0.187, 0],
    [52, 53, 0.0762, 0.0984, 0],
    [53, 54, 0.1878, 0.232, 0],
    [54, 55, 0.1732, 0.2265, 0],
    [11, 43, 0, 0.153, 0],
    [44, 45, 0.0624, 0.1242, 0.004],
    [40, 56, 0, 1.195, 0],
    [56, 41, 0.553, 0.549, 0],
    [56, 42, 0.2125, 0.354, 0],
    [39, 57, 0, 1.355, 0],
    [57, 56, 0.174, 0.26, 0],
    [38, 49, 0.115, 0.177, 0.003],
    [38, 48, 0.0312, 0.0482, 0],
    [9, 55, 0, 0.1205, 0],
])


def _load_profile_csv(path: Path, T: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    try:
        data = np.genfromtxt(path, delimiter=",", dtype=np.float32, max_rows=T, skip_header=1)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse CSV profile {path}: {exc}") from exc
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim == 0:
        raise ValueError(f"CSV profile {path} is empty")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    rows, cols = data.shape
    if cols == 0:
        raise ValueError(f"CSV profile {path} has zero columns")
    if rows < T:
        pad = np.zeros((T - rows, cols))
        data = np.vstack([data, pad])
    return data


def _assign_disturbance(dist_matrix: np.ndarray, buses: List[int], profile: np.ndarray, name: str) -> None:
    if profile.shape[1] < len(buses):
        raise ValueError(
            f"{name} columns ({profile.shape[1]}) are fewer than bus count ({len(buses)})"
        )
    if profile.shape[1] > len(buses):
        profile = profile[:, : len(buses)]
    for idx, bus_id in enumerate(buses):
        dist_matrix[:, bus_id - 1] = profile[:, idx]


def _build_spanning_tree(branch: np.ndarray) -> List[Tuple[int, int]]:
    g = nx.Graph()
    for row in branch:
        fbus, tbus, r, x = int(row[0]), int(row[1]), float(row[2]), float(row[3])
        g.add_edge(fbus, tbus, weight=r + x)
    tree = nx.minimum_spanning_tree(g, weight="weight")
    # orient the tree from slack bus 1 outward
    root = 1
    edges: List[Tuple[int, int]] = []
    for u, v in nx.bfs_edges(tree, root):
        edges.append((u, v))
    return edges


def _impedance_dict(branch: np.ndarray, radial_edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[float, float]]:
    lookup: Dict[Tuple[int, int], Tuple[float, float]] = {}
    branch_map = {(int(r[0]), int(r[1])): (float(r[2]), float(r[3])) for r in branch}
    branch_map.update({(int(r[1]), int(r[0])): (float(r[2]), float(r[3])) for r in branch})
    for u, v in radial_edges:
        if (u, v) in branch_map:
            lookup[(u, v)] = branch_map[(u, v)]
        else:
            # default tiny impedance if missing (e.g., synthetic feeder edge)
            lookup[(u, v)] = (0.001, 0.001)
    return lookup


def build_case57_environment(
    kappa: int = 3,
    T: int = 50,
    v_feeder: float = 1.0,
    v_ref: float = 1.0,
) -> GridEnv:
    base_mva = baseMVA
    N_BUSES = bus.shape[0]

    # Build radial edges and impedance dict. Ensure a feeder link from virtual node 0 to bus 1.
    radial_edges = _build_spanning_tree(branch)
    if (0, 1) not in radial_edges:
        radial_edges = [(0, 1)] + radial_edges
    impedances = _impedance_dict(branch, radial_edges)
    impedances[(0, 1)] = (0.0, 0.0)

    # Remove virtual node 0 and drop buses with no incident edges; remap to contiguous ids
    bus_set = set()
    for u, v in radial_edges:
        if u != 0:
            bus_set.add(u)
        if v != 0:
            bus_set.add(v)
    kept_buses = sorted(bus_set)
    id_map = {old: idx + 1 for idx, old in enumerate(kept_buses)}

    filtered_edges = []
    for u, v in radial_edges:
        if 0 in (u, v):
            continue
        filtered_edges.append((id_map[u], id_map[v]))

    impedances_mapped = {}
    for (u, v), z in impedances.items():
        if 0 in (u, v):
            continue
        impedances_mapped[(id_map[u], id_map[v])] = z

    # Per-unit load profiles (Pd, Qd are MW/MVar). Treat loads as negative injections.
    pd_base = bus[:, 2] / base_mva  # (N,)
    qd_base = bus[:, 3] / base_mva  # (N,)

    p_matrix_full = np.zeros((T, bus.shape[0]))
    q_n_matrix_full = np.zeros((T, bus.shape[0]))

    # Disturbance profiles from CSVs
    base_dir = Path(__file__).resolve().parent
    solar_buses = [12, 45, 44, 46, 49, 47, 38, 48, 43, 41, 37, 42, 36, 40, 35, 34, 30, 32, 33, 31]
    wind_buses = [15, 14, 9, 8, 4, 6, 5, 7]
    ev_buses = [2, 17, 20, 21, 22, 23, 24]

    solar_csv = _load_profile_csv(base_dir / "solarPV.csv", T)
    wind_csv = _load_profile_csv(base_dir / "wind.csv", T)
    ev_csv = _load_profile_csv(base_dir / "ev.csv", T)

    disturbance = np.zeros((T, bus.shape[0]))
    _assign_disturbance(disturbance, solar_buses, solar_csv, "solarPV.csv")
    _assign_disturbance(disturbance, wind_buses, wind_csv, "wind.csv")
    _assign_disturbance(disturbance, ev_buses, ev_csv, "ev.csv")

    # Synthetic solar generation for all buses (simple scaled sine), mimicking build_default_environment
    hours = np.arange(T)
    daily_cycle = 0.5 * np.sin(2 * np.pi * (hours - 6) / 24) + 0.8  # shape (T,)
    pv_scale = np.linspace(0.6, 1.0, bus.shape[0])  # different peak per bus
    solar_profile = np.outer(daily_cycle, pv_scale)  # (T, N)

    for t in range(T):
        noise = 0.0  # load noise removed
        p_matrix_full[t] = solar_profile[t] - pd_base + disturbance[t]  # generation minus load plus disturbance
        q_n_matrix_full[t] = -qd_base  # no noise

    # Remap p/q matrices to kept buses only
    p_matrix = p_matrix_full[:, [b - 1 for b in kept_buses]]
    q_n_matrix = q_n_matrix_full[:, [b - 1 for b in kept_buses]]
    N_BUSES = len(kept_buses)

    # Drop node 0 from edges and adjust p/q matrices accordingly (bus indices start at 1).
    env = GridEnv(
        n_buses=N_BUSES,
        edges=filtered_edges,
        impedances=impedances_mapped,
        v_feeder=v_feeder,
        v_ref=v_ref,
        kappa=kappa,
        T=T,
        p_matrix=p_matrix,
        q_n_matrix=q_n_matrix,
    )
    # Track mapping from original bus numbers to matrix columns (0-based)
    env.bus_ids = kept_buses
    env.bus_id_map = {bus_id: idx for idx, bus_id in enumerate(kept_buses)}
    env.load_noise = 0.0
    # Regularize R,X to ensure full rank
    eps = 1e-4
    env.R = env.R + eps * np.eye(env.N)
    env.X = env.X + eps * np.eye(env.N)

    # Drop any all-zero rows/cols (unlikely after filtering, but guard anyway)
    mask = ~(np.all(np.abs(env.R) < 1e-12, axis=1) & np.all(np.abs(env.R) < 1e-12, axis=0))
    if not mask.all():
        keep_idx = [i for i, m in enumerate(mask) if m]
        id_map = {old + 1: new + 1 for new, old in enumerate(keep_idx)}
        env.N = len(keep_idx)
        env.edges = [(id_map[u], id_map[v]) for u, v in env.edges if u in id_map and v in id_map]
        env.impedances = {(id_map[u], id_map[v]): z for (u, v), z in env.impedances.items() if u in id_map and v in id_map}
        env.R = env.R[np.ix_(mask, mask)]
        env.X = env.X[np.ix_(mask, mask)]
        env.p_matrix = env.p_matrix[:, mask]
        env.q_n_matrix = env.q_n_matrix[:, mask]
        env.x = np.zeros(env.N)
        env.v_ref = np.full(env.N, env.v_ref[0] if np.ndim(env.v_ref) else env.v_ref)
        env.v_actual = env.v_ref.copy()
        env.q_c_t = np.zeros(env.N)
        env.v_par_prev = np.zeros(env.N)

    return env


if __name__ == "__main__":
    env = build_case57_environment()
    print(f"Built case57 environment: N={env.N}, edges={len(env.edges)}, T={env.T}")

