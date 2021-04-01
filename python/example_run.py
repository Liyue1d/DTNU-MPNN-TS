from my_classes import *
from utilities import *
import time
import cplex
from decimal import *
import numpy as np
import os

window = []
visited_nodes_num = [0]
visited_or_nodes_num = [0]
visited_and_nodes_num = [0]

try:
    model.load_state_dict(torch.load("save_single_2-6-out", map_location=torch.device(device)))
except (FileNotFoundError):
    print("Save Not found")


#define timeout here
timeoutDuration = 60

#Use MPNN ?
MPNNuse = True

#define DTNU here
setOfControllables = [0, 1, 2]
setOfUncontrollables = [3]
freeConstraints = [
    [[1, 0, Decimal(1), Decimal(6)]],
    [[2, 0, Decimal(7), Decimal(10)]],
    [[3, 1, Decimal(0), Decimal(2)], [3, 2, Decimal(0), Decimal(10)]]
]
contingencyLinks = {0: [3, [[Decimal(5), Decimal(20)]]]}




a = createDNTU(setOfControllables, setOfUncontrollables, freeConstraints, contingencyLinks)

start = time.time()
try:
    treeSearch(a, window, visited_nodes_num, visited_or_nodes_num, visited_and_nodes_num, 0, False, MPNNuse,
               time.time(), timeoutDuration)
except timeout_decorator.timeout_decorator.TimeoutError:
    print("Timer expired")
end = time.time()
print("Compute time: %.2f s\n"%float(end-start))

if a.truthValue == 1:
    printStrategy(a, 0)











