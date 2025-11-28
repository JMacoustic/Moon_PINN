# Meta Fourier PINN 

This is a implementation of PINN simulation of Metamaterial in time domain. Generates re-entrant honeycomb structure with triangular mesh and simulates dynamics for sinusoidal wave input. You can optimize geometry to reduce vibration of the output node.


## Installation

First, clone repository.

    git clone https://github.com/JMacoustic/Moon_PINN.git

Next, go into the cloned repo and train the model.
    
    cd path/to/cloned/repo/
    bash scripts/setup.sh

## Training

Edit the training configuration file `configs/base_config.json` as your training preference. Once you are done, run the following script at the root.

    bash scripts/dev.sh

## Visualization

Run below code to save visualization of the trained simulation in the `outputs/visuals` folder.

    bash scripts/dev.sh