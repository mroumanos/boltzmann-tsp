Model for Boltzmann TSP

The Network:
- A network created and stored within an n x m matrix, where n = # cities and m = # epochs required to travel to them (n + 1)
- Each network node contained an n x m weight matrix

Weight matrix:
- Generally, the weight matrix for a given node, (n_i,e_j) for city 'i' and epoch 'j' followed the principles here:
    > discouraged the activation of n_i in epoch, e_k where k != j (any other time epochs) with one exception: tour completion
    > discourage other node activation within the same time epoch
    > encouraged the activation of itself in order to promote at least on active node in any epoch
    > discourage long distances travelled between any two nodes
    > encourage the active node in the final epoch to activate the same city in the first (and vice versa)
- Using the perspective of one node, these principles were followed:
    > All weights were initialized to 0
    > Weights of any adjacent node (in epoch +/- 1) i is = -e^(-w)
    > All other nodes within the same epoch received a static weight for breaking the hamiltonian
    > Weights of adjacent nodes of the same city receive the same static weight
    > Self-connections for each node uses a static bias
    > If the network completes a hamiltonian, provide additional negative weight input to the energy function
    > If it breaks a hamiltonian, provide positive weight input
    
Thank you,
Michael Roumanos