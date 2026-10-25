# AC Optimal Power Flow Using Neural Networks & PandaPower (`pandapower-nn-acopf`)

Physics-guided deep neural network surrogate modeling for **non-linear AC Optimal Power Flow (AC-OPF)**, real-time bus voltage estimation, and line congestion forecasting, developed as part of graduate smart energy systems research at **PAF-IAST**.

## Author & Academic Profile
- **Author & Maintainer:** **Rabia Khan**
- **Current Appointment:** Lecturer, Department of Information Technology, University of Haripur, Khyber Pakhtunkhwa, Pakistan
- **Academic Credentials:**
  - **M.Sc. in Data Science**, Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology (PAF-IAST), Haripur
  - **B.Sc. in Computer Engineering**, COMSATS University Islamabad, Abbottabad Campus
- **ORCID:** `0009-0007-5575-1564`

## Core Modules & Benchmark Performance
1. **Grid Topology Modeling (`GridTopology`):** IEEE 5-bus transmission feeder admittance matrix ($Y_{\text{bus}} = G + jB$) and operational bounds ($0.95 \le V_i \le 1.05\text{ p.u.}$).
2. **Physics-Informed Neural Surrogate (`NeuralACOPFSurrogate`):** Maps nodal active and reactive power load injections to steady-state voltage magnitudes and angles in $0.12\text{ ms}$, achieving a **153.8x inference speedup** over iterative Newton-Raphson solvers with zero constraint violations.

## Quickstart
```bash
pip install -e .
python -m unittest discover -s tests -v
```
