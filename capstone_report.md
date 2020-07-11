# Machine Learning Engineer Nanodegree




## Capstone Project


Arnesh Sahay  
July 2020




## I. Definition




### Project Overview


Over the course of the last 30,000 years, humans have domesticated dogs and turned them into truly wonderful companions. Dogs are very diverse animals, showing considerable variation between different breeds. On an evening walk through a given neighborhood, one may encounter several dogs that bear no semblance to each other. Other breeds of dogs are so similar that it is difficult for the untrained eye to tell them apart. Included below are a few different images, see if it would be possible for the average person to determine the the breed of these dogs.


![Labrador Retrievers are recognized as having three possible coat colors: yellow, chocolate and black](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/breed_colors.png)


![The legitimacy of a fourth color, silver, is widely contested amongst breeders](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/silver_labrador.jpg)


> LTHQ et al. [Silver Labrador Retriever Facts And Controversy](https://www.labradortraininghq.com/labrador-breed-information/silver-labrador-retriever). In _Labrador Training HQ_, 2020.


![It is not easy to distinguish between the Brittany (left) and the Welsh Springer Spaniel (right) due to similarities in the patterned fur around their eyes](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/breed_similarity_01.png)


![Another pair of dogs, the Curly-coated Retriever (left) and the American Water Spaniel (right), that are difficult to tell apart from the texture of their coats](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/breed_similarity_02.png)


It may take many weeks or months for a person to learn enough about the physical attributes and unique features of different dog breeds to effectively identify them with a high degree of confidence. It would be interesting to see if a machine learning model can be trained to accomplish the same task in a matter of a few hours. The goal of this project is to use a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) to train a dog breed classifier across 133 breeds of dogs, using the [`dogImages`](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) dataset, which consists of 8,351 total images split into 133 different categories by dog breed.




### Problem Statement


In order to identify the breed of a dog, a model would first need to identify if the image even contains a dog. Once the presence of a dog in the image is confirmed, a multi-class classifier can then begin to determine which breed of dog the image contains. For the scope of this project, 133 breeds were included in the model.


In addition to classifying images of dogs by breed, it would be interesting if the model passed human images as valid input to the classifier as well and found the closest matching breed depending on the human's face. For that purpose, another model can be used to identify if the image contains a human. Once a human has been identified as being present in the image, it can be fed into the multi-class classifier to determine which of the 133 breeds is the closest match.


All in all, the solution should have three models: two binary classifiers to determine if the image contains a human or a dog, respectively, and a third multi-class classifier to identify the dog breed of the human or dog in the image.




### Metrics


Using a test set from the `dogImages` dataset, it is possible to measure accuracy by counting `true positives` identified by the model and weighing them against the total number of predictions made. This calculation is outlined below.


![Accuracy](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/confusion_matrix.png)


Accuracy is a decent measure for evaluation of machine learning models, but a far better approach is to evaluate the precision and recall of a model. Accuracy, alone, is not a very helpful metric for understanding what changed between iterations of models. For instance, if a model has very high precision but low recall, then it might be necessary to retrain the model on more data; on the other hand, if a model has good recall but lower precision, then it might be necessary to tune the features being used. Accuracy might be identical across both of the previous scenarios and does not lend very much information for improving model performance. A graphic below explains the calculation for both precision and recall.


![Precision and Recall](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/precision_recall.png)


A further calculation can be made using precision and recall to yield the F1 score, a metric which can be used to effectively compare performance between different iterations of a model.


![F1 Score](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/f1_score.png)


For the scope of this project, it will be sufficient to calculate the precision, recall, and F1 score for the CNN against a test set from the `dogImages` dataset alone. The binary classifiers used to detect dogs and humans will not be evaluated since they are both out-of-the-box models only used for binary classification.




## II. Analysis
_(approx. 2-4 pages)_




### Data Exploration


There are two datasets for this project to be used for this project - [dogImages](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip), which contains images of dogs categorized by breed, and [lfw](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip), which contains images of humans separated into folders by the individual's name. The dog images dataset has 8,351 total images split into 133 breeds, or categories, of dogs. The data is further broken down into 6,680 images for training, 835 images for validation, and 836 images for testing. The human images dataset has 13,233 total images across 5,749 people. Below are some examples of images from both datasets.


![An image of the Afghan Hound breed from the `dogImages` dataset](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/dogImages/test/002.Afghan_hound/Afghan_hound_00125.jpg)


![An image of Aaron Eckhart, who also goes by Harvey Dent in some circles, from the `lfw` dataset](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg)


As far as the image resolutions go, the data consists of varying sizes and dimensions. Since the CNN will need to be fed in standardized input, the data will need to be pre-processed prior to using it in our model. The latter sections will cover the specifics of the pre-processing techniques to be used.




### Exploratory Visualization


When picking a model to use for transfer learning to build the CNN, it would be helpful if there was a measure to gauge the effectiveness of the model before going through training. One possible way to do this would to evaluate the models on their capability to distinguish between images of humans and dogs. In the plots below, four different models have been plotted against the training set data so the visual representation of the features used to detect dogs and humans can be understood.


![VGG-19 Model](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/transfer_learning_vgg19.png)


![ResNet-50 Model](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/transfer_learning_resnet50.png)


![Inception-V3 Model](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/transfer_learning_inception_v3.png)


![Xception Model](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/transfer_learning_xception.png)


From looking at the graphs, the ResNet-50 model seems to have the most distinct separation between dog and human data points. Inception-V3 has a close but rigid line of separation, whereas Xception and VGG-19 seem to have more interspersed data points. It is worth investigating their underlying architectures and reading any relevant research papers to determine which model to go with, but this is an interesting exercise to get another perspective.




### Algorithms and Techniques


There will be a total of three models, as stated earlier. The first model, a `face_detector` will use [OpenCV's implementation of Haar feature-based cascade classifiers](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) to determine if a human face is present, returning `true` if a face is present, or `false` if a face is not present in the image. The second model, a `dog_detector` will leverage [a pre-trained VGG-16 model](https://pytorch.org/docs/master/torchvision/models.html) to identify if the images contain a dog, returning `true` if a dog is present, or `false` if a dog is not present. The third model will be a CNN trained using [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) from [a pre-trained VGG-19 model](https://pytorch.org/docs/master/torchvision/models.html).


As the focus of this project is on the CNN multi-class classifier, it would be fitting to discuss its architecture in more depth. As a background on neural networks in general, they typically consist of an input layer and an output layer with some hidden layers in between. At a very high level, the input layer is responsible for taking in data, the hidden layers apply some changes to it, and the output layer yields the result. There will actually be two CNNs built for this project, one from scratch and another using transfer learning. The goal is to understand the architecture of a CNN better by building one from scratch and then train one using transfer learning to achieve better results. Both CNNs will contain an input layer that takes in a pre-processed 3 x 256 x 256 image tensor. That tensor will then be mapped across five convolutional layers and two fully connected layers to generate a final prediction out of 133 categories.


In the CNN that will be built from scratch, the five convolutional layers will all be normalized and max-pooled, and the two fully connected layers will be configured with 50% probability of dropout to prevent overfitting. This architecture is inspired by the AlexNet model. All layers will use Rectified Linear Units (ReLUs) for the reduction in training times as documented by Nair and Hinton.


> Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. [ImageNet Classification with Deep Convolutional Neural Networks](https://www.cs.toronto.edu/~hinton/absps/imagenet.pdf). In _Proceedings of NIPS_, 2012.


> Vinod Nair and Geoffrey Hinton. [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf). In _Proceedings of ICML_, 2010.


In the CNN built using transfer learning, [a pre-trained VGG-19 model](https://pytorch.org/docs/master/torchvision/models.html) will be used to get much better results than the CNN trained from scratch. Transfer learning involves taking a pre-trained model and using its 



![VGG-19 CNN Architecture](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/vgg19_architecture.jpg)






In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_




### Benchmark


The [pre-trained VGG-16 model](https://pytorch.org/docs/master/torchvision/models.html) used as the `dog_detector` model can be used as a benchmark model for breed classification since it can also be used to identify dog breeds in images provided as input. The model performs with an accuracy of roughly 40% on a test set of 100 randomly selected images. While this performance is not stellar, it can be used as a benchmark to compare against the CNN. The goal of this project is to create a CNN that out-performs the VGG-16 model and delivers an accuracy rate of greater than 60%.




## III. Methodology
_(approx. 3-5 pages)_


### Data Pre-processing


In addition to separating the data into training, test, and validation sets, the data also needs to be pre-processed. Specifically, all images should be resized to 256 x 256 pixels and then center cropped to 224 x 224 to ensure that they are all uniform in dimensions for the tensor. Also, the red, green, and blue (RGB) color channels should be normalized so that gradient calculations performed by the neural network during training can done more consistently and efficiently. Apart from these items, the images are part of a prepared dataset so there are no abnormalities or inconsistencies that need to be addressed in data pre-processing.




### Implementation


The end-to-end functionality of this project will employ a multitude of separate components. Any input would first be run through the `face_detector` and `dog_detector` functions to determine if the image is of a person or dog. Afterwards, the image would be fed into a CNN to determine the appropriate breed to which the image should belong.


The machine learning pipeline will include:


1. Importing the datasets
2. Binary classification
    * a. Detecting humans
    * b. Detecting dogs
3. Convolutional Neural Network (CNN)
    * a. Classifying dog breeds from scratch
    * b. Classifying dog breeds using transfer learning
4. Algorithm implementation for images provided as input
5. Model performance testing


The `face_detector` function will leverage [OpenCV's implementation of Haar feature-based cascade classifiers](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) to detect human faces in images. The `dog_detector` function will utilize [a pre-trained ResNet-50 model](https://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) to detect dogs in images. The CCN will use transfer learning and extract bottleneck features from one of the following different pre-trained models available in Keras:


* VGG-19 bottleneck features
* ResNet-50 bottleneck features
* Inception bottleneck features
* Xception bottleneck features


To better describe the function of the CNN in this project, the illustration below shows an example of how CNNs handle image classification. The convolutional and max-pooling layers extract features from a provided input image. Those features are then used to perform non-linear transformations in the fully-connected layer and produce a classification result.


![CNN Schema](https://raw.githubusercontent.com/Supearnesh/ml-dog-cnn/master/img/cnn-schema.jpg)


> Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. [ImageNet Classification with Deep Convolutional Neural Networks](https://www.cs.toronto.edu/~hinton/absps/imagenet.pdf). In _Proceedings of NIPS_, 2012.


> Vinod Nair and Geoffrey Hinton. [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf). In _Proceedings of ICML_, 2010.


In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_




### Refinement


In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_




## IV. Results
_(approx. 2-3 pages)_


### Model Evaluation and Validation


In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_


### Justification


> Karen Simonyan and Andrew Zisserman. [Very Deep Convolutional Neural Network Based Image Classification Using Small Training Sample Size](https://arxiv.org/pdf/1409.1556.pdf). In _Proceedings of ICLR_, 2015.


In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_




## V. Conclusion
_(approx. 1-2 pages)_




### Free-form Visualization


In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_




### Reflection


In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_




### Improvement


In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_




-----------


**Before submitting, ask yourself. . .**


- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
