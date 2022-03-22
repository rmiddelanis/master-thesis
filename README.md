# Master Thesis Repository

This is the repository for the code, documentation and organization of my master
thesis with the title "An Evaluation of Layer Transferability for Temporal
Convolutional Networks". The thesis was written at IAV GmbH in Berlin and
submitted to Université de Rennes 1 in Rennes, France in August 2019.

## Repo Structure
The repository contains the following documents:
*  **0_Thesis:** The final master thesis document.
*  **1_Literature:** Literature that I used to write the thesis and develop
code.
*  **2_Code:** The Python code that I wrote during the time that I worked on the
thesis project.
*  **3_Presentations:** Slides of the initial topic presentation, intermediate
result presentation and final presentation / defense.

Parts of the code are based on or taken from the code accompanying the paper
*"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for
Sequence Modeling"* by *Bai et al. (2018)*.

## Minimal working example
### Preliminaries
The project was developed using Python 3.7.1 in an Anaconda environment. The following packages are
required:
*  PyTorch (tested with version 1.0.1)
*  numpy
*  pandas

Clone the repository:
```
git clone git@github.com:rmiddelanis/master-thesis.git
```

The following Python scripts are requried for the minimal working example[^1]:

```
2_Code/
      testscript.py       // test script to train a TCN model
      model.py            // model class
      utils.py            // helper functions
      preprocessing.py    // preprocessing class
      tcn.py              // original core TCN class with small adaptations. The original class can be found at https://github.com/locuslab/TCN
```

<!--
*  ``testscript.py``
*  ``model.py``
*  ``utils.py``
*  ``preprocessing.py``
*  ``tcn.py``[^1]
-->


### Dataset
The model can work with univariate and multivariate time series to be predicted.
The dataset should fulfill the following criteria:
*  available as pickled pandas dataframe ```*.p```.
*  Columns representing Features are named ```'Ist'```, columns representing
Targets are named ```'Soll'```.
*  Each row represents a time step in the series.

A synthetically generated dataset that can be used for testing is located in
the repository as ``2_Code/data/Synthetic/synthetic_data.p``.

More such synthetic datasets can be generated using the script ``2_Code/synthetic_data_generation.py``.

### Training of a TCN model
To train a model on the synthetic dataset in
``2_Code/data/Synthetic/synthetic_data.p``, run

```@bash
python testscript.py
```

This will train a model on the specified dataset. The resulting model, a loss
plot, a plot of the prediction for the test set and the used configuration will
be stored in a new experiment folder
``2_Code/experiments/[current_date_and_time]/ ``.

Parse any of the following parameters if needed (not exhaustive):

```@bash
    --cuda          # to run the model on a CPU instead of CUDA device
    --epochs        # specify the number of epochs to train
    --ksize         # specify the size of convolutional kernels
    --levels        # specify the number of residual blocks
    --lr            # set the learning rate
    --nhid          # set the number of channels
    --series_x      # select the feature series from the dataset using a string (e.g. '0011' to select the third and fourth feature series)
    --series_y      # select the target series, usage like series_x parameters
    --validseqlen   # number of last time steps to compute the loss
    --seqstepwidth  # step width for drawing training sequences
    --seq_len       # length of sequences to be generated from the dataset
    --batch_size    # number of sequences per batch_size
```


The model can be loaded in Python with
```@Python
model = torch.load(open('2_Code/experiments/[current_date_and_time]/model.pt', 'rb'))
```
