Notebooks to train and reproduce results in [Normalizing Flows for Random Fields in Cosmology](https://arxiv.org/pdf/2105.12024)

![Animation from base distribution to model output](https://github.com/adamrouhiainen/NormalizingFlowsCosmology/blob/main/fading.gif)

train_rnvp_Gaussian.ipynb for the Gaussian fields, train_rnvp_nonGaussian.ipynb for the non-Gaussian fields, and train_rnvp_nbody.ipynb for the N-body fields

The hyperparameters easily adjustable for each notebook are n_layers (the number of flow transformations), hidden_sizes (the depth of the CNNs), kernel_size, batch_size, and base_lr.

`priormode = "whitenoise"` is uncorrelated Gaussian prior. `priormode = "correlated"` is the correlated Gaussian prior [we introduced](https://arxiv.org/pdf/2105.12024).

The loss (and validation for N-body notebook) is printed after every N_epoch number of training steps.

Requirements for all notebooks: `numpy`, `pickle`, `torch>=1.5`, `PyLab`, `scipy`, `quicklens` (provided is an updated version of quicklens for Python 3)

Additional requirement for N-body notebook: 2D Quijote density fields; instructions at https://quijote-simulations.readthedocs.io/en/latest/df.html#id1

The data is read in the N-body notebook with training_data.py, line 166. Quijote patches should be saved in `data/patches/`, or change fname_train and fname_validation depending on how Quijote data is saved.
