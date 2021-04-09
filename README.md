# Neural Symbol Arts
## *PLEASE NOTE*

I am giving up on this project for now. It seemed like a good idea but the network can't train well enough given even a moderate size to a level that it can create even the most simple symbol arts. This is here just for documentation sake. Sorry :(


## What is it?
Phantasy Star Online 2 allows you to generate Symbol Arts (fancy word for art) by layering a large number of primitive shapes together to make something more complex. Even though this sounds simple, not all of us are artistically inclined and so making our own symbol arts would be too much time and effort. This project is meant to automate the process by training a neural network model to do this for you.

## Nuts and bolts
The model is an encoder-decoder model that uses a [vision transformer](https://arxiv.org/abs/2010.11929) for the encoder and a vanilla decoder.