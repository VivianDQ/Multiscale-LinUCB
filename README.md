> 📋 This is the README.md for code accompanying the following paper

# Multiscale Non-stationary Stochastic Bandits

This repository is the official implementation of [Multiscale Non-stationary Stochastic Bandits]. 

## Dependencies

To run the code, you will need 

```
Python3, NumPy, Matplotlib
```

## Simulation

To get the results of simulations in 3 scenarios in the paper, run the following command respectively:

```
python3 ./run_contextual.py -s dim2 -k 2 -d 2
```

```
python3 ./run_contextual.py -s fix -k 100 -d 50 -cp 10
```

```
python3 ./run_contextual.py -s random -k 100 -d 10 -cp 10
```


## Results

Our implementations will automatically create a ``result/contextual`` folder. 


## Plots

To produce the same plots as in our paper, run the following command, it will create a ``plots`` folder and the figures will be saved there.

```
python3 plot_all.py
```


