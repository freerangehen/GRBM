# GRBM
Restricted Boltzmann Machine with real visible and binary hidden units. Code is written in Python 2.7 with theano. Development 
tools I relied on [ipython notebook]() running on Ubuntu Linux 14.04. 

#setting up the environment
This project was done 99% on ipython notebook with python 2.7 kernell and Theano 0.7. I followed Markus Beissinger's [blog](http://markus.com/install-theano-on-aws/) for installing Theano. The code is tested with cuda 7.0. 

Once cuda is setted up, you may install ipython/jupyter [notebook](http://randyzwitch.com/ipython-notebook-amazon-ec2/). I find it very convenient to try out smaller scripts alongside the main code.

To test the network, MIT Center for Biological and Computation Learning (CBCL) image database can be downloaded [here](http://cbcl.mit.edu/projects/cbcl/software-datasets/FaceData1Readme.html). Change path accordingly in GRBM_main.py pointing to your training example directory:

```
path='./train/face'
```


    r_a = 0.001 # training rate for a
    r_b = 0.001 # training rate for b
    r_omg = 0.00005 # trianing rate for omega 
    r_sig = 0.00005 # training rate for sigma


#example usage
The RBM is implemented in the `RBMrv_T` class in GRBM.py with constructor:

```
_init__(self, noOfVisibleUnits, noOfHiddenUnits, CD_n, aRate, bRate, omegaRate, sigmaRate,...)
```

`noOfVisibleUnits`, `noOfHiddenUnits` needs to be perfect squares for the current implementation. `CD_n` specifies the number of iterations in the Markov Chain Monte-Carlo simulations for gradient updates. `aRate`, `bRate`, `omegaRate`, `sigmaRate` specifies respective training rates for the various network parameters. The real valued RBM is known to be sensitive to training rates. For the CBCL database reasonable results were achieved by setting `aRate=0.001`, `bRate=0.001`, `r_omg=0.00005`, `r_sig=0.00005`. For other training data other update rates might work better. Other optional parameters exits including ones to control rprop gradient updates, which are not used with RMSprop working with mini-batch training. The optional parameter `sparseTargetp` is the target firing probability of the hidden units to introduce sparcity in the cost function. It is set to 0.01 by default.

A `main()` can be found as an example in GRBM_main.py. The CBCL image dateset are used as an example for unsupervised feature learning.  

The network can be trained with other data and it expects input training data as a numpy array with shape = [#example, #elment in example] such that each row is an example. Normalising inputs to the range [0,1] helps the training. 

To train the network, we call the following after instantiating an `RBMrv_T` object:

```
myRBMrv.trainMB(example_numpy_array, noOfEpoch, miniBatchSize)
```

The current version of `trainMB()` does not internally randomise the mini-batch so the input example array are expected to have been randomised already. `example_numpy_array` will be broken down into batches of `miniBatchSize` rows sequentially to compute a series of gradient updates. A proper way to carry out stochastic gradient descent is to randomise the input example array and call `myRBMrv.trainMB(example_numpy_array, 1, miniBatchSize)` in between randomisations of the input. 

A trained network can be saved and recalled by executing the following:

```
myRBMrv.saveParameters('file_name')
myRBMrv.loadParameters('file_name')
```

These are instance methods so make sure the loaded network parameters fits the dimensions of the instantiated myRBMrv object.

To sample from the RBM, use:

```
myRBMrv.genSamples(noOfgenSamples, sampleSkip)
```

`sampleSkip` is the number of MCMC iterations between the samples. `genSamples()` internally generates `noOfgenSamples + 1` examples and discard the first when returning. 




#license info
Distrubuted by Henry Ip through the MIT License
