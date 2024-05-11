# Hidden Markov Model in Human Activity (Standing, Walking, Running) Classification

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage and Structure](#usage-and-structure)
- [Repository Structure](#repository-structure)


## Introduction

This repository hosts the code for the project of Pattern Recognition and Machine Learning course. We use HMM to classify the accelerometer data sequence (3 axis) recorded from three different human activities (Standing, Walking, Running). The best approach achieves **98.89% ** classification accuracy in validation data.


## Installation

To set up the project environment, clone this repository and install the required dependencies:

```bash
git clone https://github.com/cloudisyzy/EQ2341-Hidden-Markov-Model.git
pip install -r requirements.txt
```

## Usage and Structure

This is a small project so I primarily use Jupyter Notebooks (.ipynb) to conduct simulations. If you are unfarmiliar with this, you can also convert the notebooks to .py files using
```bash
jupyter nbconvert --to script *.ipynb
```
But I cannot guarantee all scripts can be executed with no error in .py format

`train.ipynb`: Train HMM using training data and EM, some key features of trained HMM are displayed.
`test_classification.ipynb`: Use the trained HMM to classify short samples from the three classes. The short samples are not involved in training. Plot the classification accuracy and confusion matrix
`test_state_prediction.ipynb`: Use the trained HMM to predict the state sequence of two long samples, which are also not involved in training. Plot the predicted state sequences.
`visualize_data.ipynb`: Display the accelerometer data in 2D and 3D fashion. Readers can ignore this script if error arises, possibly due to javascript configurations.

For all .ipynb files, you don't need to change any variables, just click RUN.

`data`: Data collected by the authors are stored in this folder. `train.csv` is used to train HMM. `test_1.csv` and `test_2.csv` are used in validating the accuracy of predicted state of long sequences. 
	`data/running_test`