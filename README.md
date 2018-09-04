# Project work on Intelligent System: Transprecision computing through Deep Neural Network (Python)

## About

This activity is based on transprecision computing problem and it points to use a deep neural network in order to learn how to approximate a calculation. 
Not only does this project aim to use a deep neural network for predicting a discrete distribution, but it also compares different approaches of deep neural network: DeCNN (Deconvolutional),CNN and Fully Connected.

### What is Transprecision computing?

Transprecision Computing aims to abandon the abstraction of "guaranteed numerical precision" and replace it with one
more flexible and efficient. Indeed, the Transprecision Computing, in addition to the already known numerical approximation search, includes a study of mathematical theory and underlying physical principles of the computational approach, as well as the design of algorithms, customized software circuits and tools for the construction of a complex system capable of being used in different applications.
The goal, in fact, is to create a versatile system, which has the potential to be exploited in a wide variety of real problems, even not limited by computing power, that can benefit from the approximation and accuracy of the method itself. The Transprecision Computing, collects know-how related to different branches of knowledge, from programming to material sciences, in order to obtain a system that is not only approximate, but also flexible, robust and able to learn.

[For further details (ita)](http://amslaurea.unibo.it/14479/1/Utilizzo%20di%20metodi%20di%20configurazione%20automatica%20per%20un%E2%80%99applicazione%20di%20trans-precision%20computing%20su%20piattaforma%20PULP.pdf)

## Understanding the dataset

The given Dataset has been generated with the follow meaning and structure: 



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>iter</th>
      <th>in0</th>
      <th>in1</th>
      <th>y_0</th>
      <th>y_1</th>
      <th>y_2</th>
      <th>y_3</th>
      <th>y_4</th>
      <th>y_5</th>
      <th>...</th>
      <th>e_55</th>
      <th>e_56</th>
      <th>e_57</th>
      <th>e_58</th>
      <th>e_59</th>
      <th>e_60</th>
      <th>e_61</th>
      <th>e_62</th>
      <th>e_63</th>
      <th>e_64</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>float16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>float16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>float16</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>float16</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>float16</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 134 columns</p>
</div>

All the data has to be seen as a discrete distribution where each bin-th reppresents the probability that the number we are considering for the operation (sum,difference,moltiplication,...) is in the range defined by the bin-th. Bins are equally spaced in logarithmic scale. The left-most and right-most bins are bounded by infinities, and the middle bin is reserved for values that are close to zero. For this reason, the number of bins should always be odd. Basically, it has to be seen as the follow image:
![example](/image/example.png) 

However, in order to have a more compact dataset, the input (in0 and in1) is an integer representing the index of the bin that has to be set as 1. The reason why we can set the bin as 1 is because it is an exact discrete distribution: we are sure about the input, so the probability is 1. This is useful because we can easly hot-one encode the input.
On the contrary, the y_i values are not exact, therefore every bins is a value from 0 to 1 (needed to be normalized) and the sum of all the (y_i-th) bins must be 1. The same idea has been done for the error values. 
Different precision's levels can be chosen: float16,float32 or float64. 
Furthemore, it is also possible to choose the number of bins we desire (65 by default) and the
number of iterations of random data generation. 
The iterations consists in filling each bin in the discretization with "random-samples-bin" values, picked uniformly at random within the bin intervals for all but the left-most and right-most bins (since one of their boundaries is not finite). Samples for these bins are obtained by replaing the infinite with finite ones.



## Fully Connected 

In this case, it has been used: the sum, float32 precision's level and iteration=3 (at data-generation time). 
As you can seen in the Architecture section, the input shape is (,130,1). 

Download [here](/transprecison/transpe_FC.ipynb) the jupyter file.

### Architecture 
![fully connected](/image/fc1.png) 


## CNN

In this case, it has been used: the sum, float32 precision's level and iteration=3 (at data-generation time). 
As you can seen in the Architecture section, the input shape is (,130,1). 

Download [here](/transprecison/transpe_CNN.ipynb) the jupyter file.

### Architecture 
![CNN](/image/cnn.png) 


## DeCNN

A deconvolutional neural network is a neural network that performs an inverse convolution model. Basically, we are “running a CNN backward” but the actual mechanics of deconvolutional neural networks are much more sophisticated than that. We are adding patterns to our data, it is slightly similar to unpooling.

Download [here](/transprecison/transpe_DeCNN.ipynb) the jupyter file.

### Architecture 
![DeCNN](/image/decnn.png) 


## Results and comparison between the architectures 

![DeCNN](/image/result1.png) 
![DeCNN](/image/result2.png) 

As we can see the results are very good in all the architectures, however, some consideration has to be made:
* The fully connected Network works perfectly and is way faster than the others.
* The Deconvolutional Network is definitely slower compared to the FC, however, according to what we expected, it takes less epochs to get the desired categorical accuracy; so we could run less epochs and get the same results.
* The Convolutional Network is the worst of them, it takes longer, due to the convolutional operations and the more layers. Furthermore, the trend of the categorical accuracy over epochs does not allow us to save training time.


## Requirements & references  : 

* numpy
* pandas
* [sklearn](http://scikit-learn.org/stable/)
* [keras](https://keras.io/)
* [tensorflow](https://www.tensorflow.org/)
* [Python 3](https://www.python.org/download/releases/3.0/)
* [Jupyter Notebook (Suggested)](http://jupyter.org/)
* [Randomness in Deconvolutional Networks for Visual Representation](https://arxiv.org/pdf/1704.00330.pdf)


