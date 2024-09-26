# Actuator-Neural-Net

## Info
This is a torque estimation framework for dynamic robotic actuators. There are three versions of the proposed torque estimation framework: position velocity acceleration-GRU (PVA-GRU [main]), position velocity-GRU (PV-GRU [pv-gru_net]), and multi-layer perceptron (MLP [mlp_net]). The MLP version is based on the Actuator Net proposed by ETH Zurich.
This framework was included in a paper submitted to ICRA 2025. The corresponding video submission can be accessed via the following link: https://youtu.be/SJexZHEwWtc

## User-Manual
- Please do not edit main. For development on this repo, use branches and pull requests. 
- For use in other projects, create a submodule inside another repository.

### Setting up as submodule
Enter this command into your own repo
```bash
git submodule add https://github.com/alvister88/Actuator-Neural-Net.git <path/to/submodule>
```
```bash
git submodule update --init --recursive
```

### Installation
This library is pip installable
```bash
pip install actuator-nn
```
#### Dev Install
Run the command inside the root directory of the repository after enabling the python environment.
```bash
pip install -e .
```
