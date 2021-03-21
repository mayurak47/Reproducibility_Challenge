# Reproducibility_Challenge
Code for reproducing the paper "Neural Networks Fail to Learn Periodic Functions and How to Fix It" as part of the ML Reproducibility Challenge

Due to the interactive nature of the experiments, most implementations are provided as Jupyter notebooks.

<h3> Description </h3>

```Extrapolation_experiments.ipynb``` contains the basic extrapolation experiments on analytical functions with neural networks having different nonlinearities.

In ```Snake_simple_experiments.ipynb```, the *snake* activation function is visualized, it is shown that *snake* can regress *sin(x)* and an MLP is trained on MNIST demonstrating the optimization capability of *snake*.

```Snake_applications.ipynb``` contains the main experiments of the paper - training a ResNet18 with *snake* on CIFAR-10, the various regression experiments, and a comparison of a feedforward *snake* network and a ReLU RNN.

```Sinusoid_a_comparison.ipynb``` demonstrates how different *a* influences learning.

```dcgan.py``` implements a DCGAN on the MNIST dataset, using the specified nonlinearity in the generator and discriminator networks.

```Sentiment_Analysis.ipynb``` is an attempt at using an LSTM network with the *snake* activation for sentiment analysis on the IMDB Movie Reviews Dataset.


<h3> Usage </h3>

Clone the repository with ```git clone https://github.com/mayurak47/Reproducibility_Challenge```.

Preferably create a Python virtual environment (e.g. ```conda create -n <env_name> python=3.6.9```) and activate it. Install the necessary libraries with ```pip install -r requirements.txt```. In case of any inconsistencies or errors, please install the appropriate version of the following packages manually, using ```pip``` or ```conda```:

```
numpy
torch
torchvision
matplotlib
scikit-learn
tqdm
jupyter
pandas
nltk
```


Run the notebooks, if necessary modifying any (hyper)parameters in the relevant files in ```data```, ```models```, ```utils.py```, or in the notebook itself.

For the GAN experiment, run the commands ```python dcgan.py --nonlinearity=snake``` and ```python dcgan.py --nonlinearity=leakyrelu```.
