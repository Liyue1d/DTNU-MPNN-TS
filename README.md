# DTNU-MPNN-TS

Computes a Restricted-Time based Dynamic Controllability (R-TDC) for Disjunctive Temporal Networks with Uncertainty.

Required programs:
- Python3
- CPLEX

Required python3 librairies:
- CPLEX python
- Docplex
- Pytorch
- Pytorch-geometric
- decimal
- timeout
- numpy
- timeout_decorator

Instructions:

1) Install the required libraires, and ensure a working CPLEX environment is installed and Docplex has access to it.

2) Edit the file example_run.py to change the DTNU to the one desired. Current DTNU is from Cimatti et al's paper.

3) Run example_run.py. The pre-trained MPNN is used to guide the tree search, and a R-TDC strategy is displayed if one is found.
