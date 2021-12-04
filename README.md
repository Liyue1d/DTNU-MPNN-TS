# DTNU-MPNN-TS

Computes a Restricted-Time based Dynamic Controllability (R-TDC) for Disjunctive Temporal Networks with Uncertainty (DTNU).
A DTNU is of the form: (A,U,C,L) where:
- A is a set of controllable variables
- U is a set of uncontrollable variables
- C is a set of contraints, with possible disjunctions
- L is a set of contigency links, with possible disjunctions

Uses a pre-trained graph neural network to guide a tree search algorithm to search for R-TDC strategies.


Required programs:
- Python3
- CPLEX
- CUDA (optional, for GPU inference)

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

3) Run example_run.py. The pre-trained MPNN is used to guide the tree search (the file save_single_2-6-out includes the pre-trained weights that were found following the training in the paper), and a R-TDC strategy is displayed if one is found.



-----------------------------------------

BENCHMARKS

File 10-20.txt containts 500 DTNUs. Each DTNUs has between 10 to 20 controllables and 1 to 3 uncontrollables.

File 20-25.txt containts 500 DTNUs. Each DTNUs has between 20 to 25 controllables and 1 to 3 uncontrollables.

File 25-30.txt containts 500 DTNUs. Each DTNUs has between 25 to 30 controllables and 1 to 3 uncontrollables.

Each DTNU is written as:

-Set of controllables: A list which contains the ID of each controllable timepoint.

-Set of uncontrollables: A list which contains the ID of each unontrollable timepoint.

-Set of free constraints: A list which contains 'disjuncts' (i.e. A disjunct refers to several contraints linked with OR operators). A disjunct is a list which contains several 'conjuncts'. A conjunct is a list of the form: Either [timepoint1, timepoint2, lower_bound, upper_bound], and represents the constraint timepoint1 - timepoint2 \in [lower_bound, upper_bound] Or either [timepoint, lower_bound, upper_bound] and represents the constraint timepoint \in [lower_bound, upper_bound]

-Set of contigency links: A dictionnary with multiple entry. Each key of the dictionnary is a controllable timepoint which triggers the activation of an uncontrollable timepoint. The associated value of the key is a list which containts, as first element, the ID of the uncontrollable timepoint which gets triggered, and as second element a list of time bounds, each of which the activated uncontrollable timepoint may occur in.

