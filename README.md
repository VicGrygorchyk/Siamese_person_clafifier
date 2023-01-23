# Siamese NN implementation
Pytorch Siamese NN based on RetNet [example](https://github.com/pytorch/examples/tree/main/siamese_network).<br/>
It compares two images (mostly persons photos) to classify them as similar or not (the same photo or not).

## Dataset
Labeling done by Active learning:
1) First, created manually 100 examples
2) Train the model (overfiting is okay)
3) Get the wrong predictions 