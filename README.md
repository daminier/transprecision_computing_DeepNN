# Project work on Intelligent System: Transprecision computing through Deep Neural Network

## About

Not only aims this project to use a deep neural network for predicting a discrete distribution, but it also compares different approaches of deep neural network: CNN and Fully Connected.

### What is Transprecision computing?

## Dataset

The given Dataset has been generated through the follow meaning and structure: 



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

All the data has to be seen as a discrete distribution reppresenting a probability, however, in order to have a more compact dataset the input (in0 and in1) is represented as the index of the bin that has to be set as 1. The reason why we can do that is because is an exact discrete distribution: we are sure about the input, so the probability is 1. 
This is useful because we can easly hot-one encode the input.
On the contrary, the y_i values are not exact, therefore every bins is a value from 0 to 1 (needed to be normalized) and the sum of all the (choosen sample) bins must be 1. 
The same idea has been done for the error values and the different precision's levels can be chosen: float16,float32 or float64. Furthemore, it is also possible to choose the number of bins we desire (65 by default). 

## Example

Please download the jupyter file in the main repository or alternatively the other formats.

## References 
