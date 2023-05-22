# DeepBern-Nets: Taming the Complexity of Certifying Neural Networks using Bernstein Polynomial Activations and Precise Bound Propagation
DeepBern-Nets is class of Neural Networks with Bernstein Polynomial based activation functions. Bernstein polynomials exhibit several key properties that allows tight computation of output bounds of all neurons in the network making them highly amenable to certification tasks. More details can be found in the paper [DeepBern-Nets: Taming the Complexity of Certifying Neural Networks using Bernstein Polynomial Activations and Precise Bound Propagation](arxiv).

# Installation
We recommend using the provided conda environment file to install all dependencies. To create a conda environment, run

```
conda env create --name deepbern --file=environment.yml
conda activate deepbern
```

Make sure the environment was created without errors.

# Training 
The main training file is `train.py` which accepts a `YAMLe` configuration file. For example, to run the code using one of the example configs, run

`python train.py configs/mnist_FCNNb_clean.yml`

There are 3 supported training modes: 

1.    Certified training mode (using Bern-IBP bounding)
2.    Adversarial training mode (using PGD attacks)
3.    Clean training mode

The training mode can be specified by modifying the `TRAIN.MODE` parameter in the configuration file. To see examples of different training modes and their usage, refer to the sample configuration files located in the `configs` directory. The models are saved in the `experiments` folder which can be configure in the config file.




