# Neural Symbol Arts
## *PLEASE NOTE*

This branch is dedicated to training each model in the long, generative way (one layer at a time while accumulating gradients then after predicting each 225 layers one at the time we backprop). This is slower and FAR less memory friendly but it SHOULD produce better results (maybe....)

## What is it?
Phantasy Star Online 2 allows you to generate Symbol Arts (fancy word for art) by layering a large number of primitive shapes together to make something more complex. Even though this sounds simple, not all of us are artistically inclined and so making our own symbol arts would be too much time and effort. This project is meant to automate the process by training a neural network model to do this for you.

## Nuts and bolts
The model is an encoder-decoder model that uses a [vision transformer](https://arxiv.org/abs/2010.11929) for the encoder and a vanilla decoder. The model is trained on predicting the next layer (like next sentence prediction).

## Things that could be done
- Change the training scheme
    * Pretrain encoder using masked patch prediction, BYOL or some sort of contrastive method
    * Train the decoder using masked layer prediction (ala BERT) or some sort of generative adversarial method
- Tweak the parameters
    * Learning rates, optimizers, transformer depths, transformer heads, whatever
- Blow the whole thing up
    * I dunno, the architecture is pretty vanilla so maybe something more outside the box would work better?
