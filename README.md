# tframe-light
 This is a tensorflow 2 wrapper where the ideology is from original [tframe](https://github.com/WilliamRo/tframe)
 
## What is the difference?
* In tframe, layers should be strictly from the Layer classes. In tframe-light, as long as the function is callable, it can be regarded as a layer.
* It supports multi-input and multi-output DAG.
* Taking the advantages of the eager model from tensorflow 2, tframe-light supports dynamic computational graphs :), while losses some speed :( . 
* In tframe, only the layers such as Conv2D and ReLU are from the original tensorflow. In tframe-light, we use the keras model using its functional api. This implies, the model can be easily loaded in antoher scipts and we do not need to build networks again. Besides, we can enjoy some convenient functions from keras :), such as model.summary() and model.trainable_variables() It can be less cutomized but much easier.
* Since we use the functional api to build keras models, any deep learning codes using tensorflow2 online can be direcly replemented by ctrl-c and ctrl-v :p. I am not encouraging anyone to copy codes :) I mean we got much more resources to learn.

## The following tframe's functions has been completed in tframe-light
* Network Searching
* Gather experiments to the note and visualize them.

## To do list
* Visualize features map
* Visualize kernel
* Visualize heatmap
* RNN, RL and GAN
* Support both the eager mode and graph mode
* Code reformatting (Long term)
* Demo codes
