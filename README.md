# Multi-Source Predictive Distributed Voltage Control (TPWRS Submission)

This repository provides a example scripts from the paper _“Multi-Source Predictive Distributed Voltage Control”_. Data files will be uploaded later; paths can be adjusted in the scripts as needed.

## Files in Scope
- `voltage_control.py` — Main driver for distributed voltage control. It wires prediction, control (OCL vs. incremental/nominal), and evaluation/plots. Depends on an environment definition and local data files.
- `case57_env.py` — Modified 57-bus grid environment with line parameters, node mapping, and disturbance loading/generation utilities for quick experiments.

## Dependencies
- Python 3.8+
- Core libs: `numpy`, `scipy`/`control`, `torch`, `scikit-learn`, `matplotlib`
- Data: CSV profiles for disturbances (e.g., load/PV). These will be provided later.

## Quick Start
1) Install requirements (example):
   ```bash
   pip install -r requirements.txt
   ```
2) Place the forthcoming CSV data in the expected paths (or update the paths in the scripts).
3) Run the controller example:
   ```bash
   python voltage_control.py
   ```
   This will build the environment (e.g., via `case5_env.py`), execute control rollouts, and print/plot key metrics.

## Notes
- The code is a concise, self-contained pipeline for the method of TPWRS submission.
- For larger networks or custom data, extend the environment definition (mirroring `case5_env.py`) and adjust the driver accordingly.


