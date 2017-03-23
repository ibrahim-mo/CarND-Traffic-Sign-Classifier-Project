#**Traffic Sign Recognition** 

##Proj 2 in the udacity CarND program
###Ibrahim Almohandes, 22-Mar-2017

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data-vis.png "Data Visualization"
[image2]: ./test_images/14.jpg "Traffic Sign 14"
[image3]: ./test_images/1.jpg "Traffic Sign 1"
[image4]: ./test_images/40.jpg "Traffic Sign 40"
[image5]: ./test_images/12.jpg "Traffic Sign 12"
[image6]: ./test_images/11.jpg "Traffic Sign 11"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ibrahim-mo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in code cell [2] of the IPython notebook.  

I used the *len()* function and the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in code cell [3] of the IPython notebook.

Here is an exploratory visualization of the data set. First, I showed a random image from each of the 43 unique classes/lables. Then, I drew a bar chart showing how the data classes are distributed (bin count of each class/label).

![Data Visualization][image1]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in code cell [6] of the IPython notebook.

I normalized the features for each of the training, validation, and test sets using Min/Max scaling by converting the RGB values [0, 255] into the [0.1, 0.9] range. I normalized the image data because large feature values could overflow at any of the next layers, and hence destabilize the training process.

Note: I tried first to convert the images into grayscale, but I got very poor performance during training. I think the reason is that there's important information in the colors of the traffic signs that a successful training process relies on (in addition to other charcteristics like shape). This is different from other cases when there's very little info added by the color features.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data imported from the pickled files in code cell [1] of the IPython notebook is already pre-split into training, validation, and test sets. From the numbers reported in code cell [2], I notice that 24% of all data went into test set, then 11% of the remaining data went into validation set. The rest represent the training set.

A common splitting technique is to first randomly sample 10-20% of all data as the test set, then randomly split the remaining data into training and validation sets (such as 9:1 or 8:2 ratio).

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in code cell [7] of the IPython notebook. 

My final model consisted of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x3 RGB image                             |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x16                 |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x40   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x40                   |
| Flattenning           | 5x5x40 = 1000                                 |
| Fully connected       | Linear, outputs 250                           |
| RELU                  |                                               |
| Fully connected       | Linear, final output of 43 uniques classes    |


This architecture is based on the well known LeNet-5. However, after lots of experimentation with the layers and their input/output dimensions, I found that dropping one of the full (linear) layers still achieved a similarly good validation and test performance (even slightly better). So my model has two convolutional and two full (linear) layers, plus soft pooling and RELU activation in-between.

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in code cell [9] of the IPython notebook. 

To train the model, I first shuffled the training data. Then, I applied an iterative process with the following parameters: batch size of 128, and number of epochs (iterations) as 50.

For the gradient descent algorithm, I applied the well known Adam (Adaptive Moment Estimation) Optimizer with a learning rate of 0.001.

I experimented with different values of batch size, number of epochs, and learning rate to achieve a satisfactory validation and test performance.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in code cell [9] of the Ipython notebook.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.951
* test set accuracy of 0.937

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
I chose the LeNet-5 because it is a widely used architecture in deep learning esp. for processing image data.

* What were some problems with the initial architecture?
Yes. When I converted the images into grayscale, the performance was very poor, hence I decided to remove this step from data prepeartion.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I experimented with the layers by removing one of the intermediate full layers as well as trying different input and output dimensions. I found that removing one full (linear) layer achieved a slightly better performance. That means it was slightly overfit when we used all the layers.

* Which parameters were tuned? How were they adjusted and why?
Different sets of input and output Dimensions for the layers, in order to achieve the best combination of validation and test performances. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
One of the decisions I made was to run the IPythone notebook on a GPU instance on AWS which was much faster than my local CPU. The convolutional layers in the architecture allow gradual (or deep) learning price that starts from very fine features (like lines and color blobs) all the way to full-blown features (like faces and shapes). And this is a fundamental basis of deep learning.

If a well known architecture was chosen:

* What architecture was chosen? LeNet 5 with a slight modification (2 covolutional and 3 full layers, as well as intermediate Relu and max pooling).
* Why did you believe it would be relevant to the traffic sign application? Sophisticated image processing applications (like this one) can benefit from this well know architecture for deep learning.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The training accuracy was an ideal 100% (I assume because I didn't deal with sophisticated problems like rotation, blur, dark light and and others), plus the training set was not very big). The validation and test accuracy were high (more than 93%)  with a small degradation in the test compared to validation. I didn't calculate training performance, as I didn't think it was necessary for this assignment.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I downloaded from the web:

![Traffic Sign 14][image2] ![Traffic Sign 1][image3] ![Traffic Sign 40][image4] 
![Traffic Sign 12][image5] ![Traffic Sign 11][image6] 

You can see the code for loading and normalizing these images in code cell [16].

The first image,  a stop sign, was not classified correctly because the sign is not centered in the image and/or because the picture was taken at night (dark). In fact, when I downloaded another stop sign image that was centered and taken during daytime (bright), it was classified correctly.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in code cell [17] of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop					| Priority road					 				|
| 30 km/h				| 30 km/h      									|
| Roundabout			| Roundabout									|
| Priority road			| Priority road									|
| Right of way			| Right of way									|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%, as shown in code cell [18]. This compares somewhat favorably to the accuracy on the test set of 93.7%. One reason for this lower performance is that 5 images are a small number to test with (for raw web images). A second reason is that the original messages are in jpeg format which has a distortion factor compared to, say, PNG images. I also mentioned issues like different lighting and blueness situations as well as position and rotation (which we didn't train much on).

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in code cell [27] of the Ipython notebook.

For the first image, the model is very sure that this is a *priority road* sign (probability of almost 1.0), while in fact its as a *stop sign*. The top five soft max probabilities were (approximately):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00    				| Priority road   								| 
| 1.00     				| 30 km/h 										|
| 1.00					| Roundabout									|
| 1.00	      			| Priority road					 				|
| 1.00				    | Right of way      							|

As for the probability of the first image being a stop sign, it's very small (almost zero) and it came 5th in the top 5 list! For a possible explanation, please refer to my earlier discussion.
