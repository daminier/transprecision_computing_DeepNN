# Project work on Intelligent System: Transprecision computing through Deep Neural Network

## About

Not only aims this project to use a deep neural network for predicting a discrete distribution, but it also compares different approaches of deep neural network: CNN and Fully Connected.

### What is Transprecision computing?

## Dataset

The given Dataset has been generated through the follow meaning and structure: 
![image](/images/dataset.png)
All the data has to be seen as a discrete distribution reppresenting a probability, however, in order to have a more compact dataset the input (in0 and in1) is represented as the index of the bin that has to be set as 1. The reason why we can do that is because is an exact discrete distribution: we are sure about the input, so the probability is 1. 
This is useful because we can easly hot-one encode the input.
On the contrary, the y_i values are not exact, therefore every bins is a value from 0 to 1 (needed to be normalized) and the sum of all the (choosen sample) bins must be 1. 
The same idea has been done for the error values and the different precision's levels can be chosen: float16,float32 or float64.

## Example

Please download the jupyter file in the main repository or alternatively the other formats.

## References 
