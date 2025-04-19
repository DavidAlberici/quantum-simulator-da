# Requirimientos 
- Python 3.12+  
- Packages: `pip install -r requirements.txt` 

# General information
The code is mostly done in files qregistry and qdensity. Use examples are included in main.py, and unit tests 
(that could also count as examples) are included in test_qdensity and test_qregistry.py files. Finally, there is a 
markdown file computational_cost.md, in which the computational cost of key functions is calculated.

## QRegistry and QDensity
Both classes are the core of this project. They are quantum computer simulators (gate model). QRegistry uses a 
statevector internally to be able to work, QDensity a density matrix.

## Use examples and tests
Class main.py contains examples, when you execute it, you also get prints indicating what is happening. 
The test_qregistry and test_qdensity classes contain unit tests. There are some performance tests as well, but they
do not fail or succeed, they just print the "time it took" to make the computation.

## Computational cost
The file computational_cost.md show the calculation of the computational cost of 3 key functions of the statevector 
simulator. (1) initializing the system, (2) applying a gate and (3) measuring a qbit.
