# Boilerplate_code
My boilerplate code for lots of usual Pytorch stuff

1- "Regression_NN_GP":
    A) Neural Net
    - set up Cuda
    - Standard scaling via scikit
    - Dataset, Dataloader classes and examples
    - Neural Net architecture (MLP)
    - Early Stopping class
    - Train_step, Test_step, training set up and run
    B) Gaussian Process regressor on GPyTorch
    - Exact model, Gaussian likelihood
    - Training loop, predictions

2- "Fourier Transform":
    - use of scipy.fft.rftt to compute the complex coefficients of a Fourier transform
    - use of scipy.fft.rfftfreq to compute the corresponding frequencies
    - plot the Fourier transform of a signal
    - use of scipy.signal.welch to get the spectral density of a signal