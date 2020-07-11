# Classification of Dog Breeds using a Convolutional Neural Network (CNN)




## Table Of Contents


- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
  * [Log in to the AWS console and create a notebook instance](#log-in-to-the-aws-console-and-create-a-notebook-instance)
  * [Use git to clone the repository into the notebook instance](#use-git-to-clone-the-repository-into-the-notebook-instance)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  * [Step 1 - Importing the datasets](#step-1---importing-the-datasets)
  * [Step 2 - Binary classification](#step-2---binary-classification)
    + [Part A - Detecting humans](#part-a---detecting-humans)
    + [Part B - Detecting dogs](#part-b---detecting-dogs)
  * [Step 3 - Convolutional Neural Networks (CNNs)](#step-3---convolutional-neural-networks--cnns-)
    + [Part A - Classifying dog breeds from scratch](#part-a---classifying-dog-breeds-from-scratch)
    + [Part B - Classifying dog breeds using transfer learning](#part-b---classifying-dog-breeds-using-transfer-learning)
  * [Step 4 - Pipeline implementation for images provided as input](#step-4---pipeline-implementation-for-images-provided-as-input)
  * [Step 5 - Model performance testing](#step-5---model-performance-testing)
- [Important - Deleting the endpoint](#important---deleting-the-endpoint)




## Introduction


Over the course of the last 30,000 years, humans have domesticated dogs and turned them into truly wonderful companions. Dogs are very diverse animals, showing considerable variation between different breeds. On an evening walk through a given neighborhood, one may encounter several dogs that bear no semblance to each other. Other breeds of dogs are so similar that it is difficult for the untrained eye to tell them apart. Included below are a few different images, see if it would be possible for the average person to determine the breed of these dogs.


![It is not easy to distinguish between the Brittany (left) and the Welsh Springer Spaniel (right) due to similarities in the patterned fur around their eyes](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/breed_similarity_01.png)


![Another pair of dogs, the Curly-coated Retriever (left) and the American Water Spaniel (right), that are difficult to tell apart from the texture of their coats](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/breed_similarity_02.png)


It may take many weeks or months for a person to learn enough about the physical attributes and unique features of different dog breeds to effectively identify them with a high degree of confidence. It would be interesting to see if a machine learning model can be trained to accomplish the same task in a matter of a few hours. The goal of this project is to use a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) to train a dog breed classifier across 133 breeds of dogs, using the [`dogImages`](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) dataset, which consists of 8,351 total images split into 133 different categories by dog breed. Everything for this project was done on Amazon Web Services (AWS) and their SageMaker platform as the goal of this project was to further familiarize myself with the AWS ecosystem.




## Setup Instructions


The notebook in this repository is intended to be executed using Amazon's SageMaker platform and the following is a brief set of instructions on setting up a managed notebook instance using SageMaker.


### Log in to the AWS console and create a notebook instance


Log in to the AWS console and go to the SageMaker dashboard. Click on 'Create notebook instance'. The notebook name can be anything and using ml.t2.medium is a good idea as it is covered under the free tier. To enable GPUs for this particular project, it is recommended to use a ml.p2.xlarge instance. Also it is advised to increase the memory allocation to 15-25GB and to create a [Lifecycle Configuration](https://docs.aws.amazon.com/sagemaker/latest/dg/notebook-lifecycle-config.html) for any required library dependencies. For the role, creating a new role works fine. Using the default options is also okay. Important to note that you need the notebook instance to have access to S3 resources, which it does by default. In particular, any S3 bucket or object with 'sagemaker' in the name is available to the notebook.




### Use git to clone the repository into the notebook instance


Once the instance has been started and is accessible, click on 'Open Jupyter' to get to the Jupyter notebook main page. To start, clone this repository into the notebook instance.


Click on the 'new' dropdown menu and select 'terminal'. By default, the working directory of the terminal instance is the home directory, however, the Jupyter notebook hub's root directory is under 'SageMaker'. Enter the appropriate directory and clone the repository as follows.


```
cd SageMaker
git clone https://github.com/Supearnesh/ml-dog-cnn.git
exit
```


After you have finished, close the terminal window.




## Machine Learning Pipeline


This was the general outline followed for this SageMaker project:


1. Importing the datasets
2. Binary classification
    * a. Detecting humans
    * b. Detecting dogs
3. Convolutional Neural Networks (CNNs)
    * a. Classifying dog breeds from scratch
    * b. Classifying dog breeds using transfer learning
4. Algorithm implementation for images provided as input
5. Model performance testing




### Step 1 - Importing the datasets




### Step 2 - Binary classification




#### Part A - Detecting humans




#### Part B - Detecting dogs




### Step 3 - Convolutional Neural Networks (CNNs)




#### Part A - Classifying dog breeds from scratch




#### Part B - Classifying dog breeds using transfer learning




### Step 4 - Pipeline implementation for images provided as input




### Step 5 - Model performance testing




## Important - Deleting the endpoint


Always remember to shut down the model endpoint if it is no longer being used. AWS charges for the duration that an endpoint is left running, so if it is left on then there could be an unexpectedly large AWS bill.


```python
predictor.delete_endpoint()
```
