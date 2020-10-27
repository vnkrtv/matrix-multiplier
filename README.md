# Matrix multiplier

## Description 

Calculates all possible products of vectors in a given matrix with specified threshold, preserving the final product chains.  
Written on libtorch (C++ Distributions of PyTorch).

## Options

List of options:  
- ```--conditions-count, -n [integer] [required]``` - Conditions count
- ```--timestamps-count, -m [integer] [required]``` - Timestamps count
- ```--threshold, -t [number] [required]``` - Threshold
- ```--seed, -s [integer]``` - Random seed (21 by default)
- ```--output, -o [string]``` - Result file name (stdout by default)
- ```--cuda``` - Training on GPU with CUDA
- ```--map-reduce``` - Use map reduce  
- ```--recursion``` - Algorithm with recursion  
- ```--help, -h``` - Displaying help

## Installing

- Clone this repo:
  - ```git clone https://github.com/vnkrtv/matrix-multiplier.git```
- Build util:
  - ```cd matrix-multiplier```
  - ```cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch```
  - ```cmake --build . --config Release```