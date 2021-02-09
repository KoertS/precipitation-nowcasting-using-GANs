# CGAN on MNIST 

Here I apply GANs to the hand-written number dataset MNIST to gain some experience with working with GANs.
I started over as previous attempt where I based the network architecture on a [kaggle notebook on CycleGANs](https://www.kaggle.com/amyjang/monet-cyclegan-tutorial) and Ying's thesis failed. 
Now the approach is to stick to the DCGAN architecture which has shown to be more stable during training.
Start DCGAN code given in [tensorflow tutorial](https://www.tensorflow.org/tutorials/generative/dcgan). 
I added a condition channel to the architecture with the condition being the class label (number).

Things to try out: 
- Add [Shaply](https://github.com/slundberg/shap) values
- Use wandb to gain more insight in the trianing process
