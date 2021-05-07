# Neural Symbol Arts
## *Update 4/28/2021*

This has come a lot further than I initially thought it would. While the status of it's success is still "absolute garbage" I will say it has at least been a great learning experience with Pytorch, attention mechanisms and generative methods. Honestly, not too bad. Making it public again because at least the code looks okay?

## What is it?
Phantasy Star Online 2 allows you to generate Symbol Arts (fancy word for art) by layering a large number of primitive shapes together to make something more complex. The model attempts to take in a PNG of size 576x288 and convert it to a SAML (symbol art markdown language) format to easier import into PSO2

## Why is it?
Because not all of us are artistically inclined and/or have someone that can convert images for us ¯\\\_(ツ)\_/¯

## How is it?
The end to end model is an encoder-decoder model that uses a [vision transformer](https://arxiv.org/abs/2010.11929) (or some variant thereof) for the encoder and a generative decoder (right now just a vanilla with improvements or [routing transformer](https://arxiv.org/pdf/2003.05997.pdf). Data is picked and converted by hand from a number of sources.

## Who is it?
Wefor-motherfucking-got and anyone that uses it :)

## Where is it?
Here because it would bother me if it wasn't but.... um..... Github?

## Things that could be done
- Change the training scheme
    * ~~Pretrain encoder using masked patch prediction, BYOL or some sort of contrastive method~~
      + This wasn't successful and actually makes the whole script less general. I tried BYOL and MPP and both weren't helping
    * ~~Train the decoder using masked layer prediction (ala BERT) or some sort of generative adversarial method~~
      + This can't even work given that mask layer prediction is an encoder technique. Discriminator training could work but all things considered I think it isn't worth pursuing for now
    * Pretrain embeddings by taking the final hidden state of some sort of existing backbone (ala ResNet) and using that as the embeddings per layer
      + This is implemented right now with a MobileNetV3-Small backbone. Testing is being done right now with embeddings both perminately frozen and using a cold start where embeddings are unfrozen at some future epoch
- Tweak the parameters
    * Learning rates, optimizers, transformer depths, transformer heads, whatever
- Blow the whole thing up
    * ~~I dunno, the architecture is pretty vanilla so maybe something more outside the box would work better?~~
      + It got blown up once already for a refactor. Unless going full on nuclear is the way to success I am not doing that again
