# Multiclass classification model using a custom convolutional neural network in TensorFlow
> **Problem statement** : 

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

The data set contains the following diseases:

* Actinic keratosis
* Basal cell carcinoma
* Dermatofibroma
* Melanoma
* Nevus
* Pigmented benign keratosis
* Seborrheic keratosis
* Squamous cell carcinoma
* Vascular lesion
## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- The solution is divided into the following sections:
- **Project Pipeline**
    - Data Reading/Data Understanding → Defining the path for train and test images 
    - Dataset Creation→ Create train & validation dataset from the train directory with a batch size of 32. Also, make  sure you resize your images to 180*180.
    - Dataset visualisation → Create a code to visualize one instance of all the nine classes present in the dataset 
    - Model Building & training : 
        -Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale - images to normalize pixel values between (0,1).
        - Choose an appropriate optimiser and loss function for model training
        - Train the model for ~20 epochs
        - Write your findings after the model fit. You must check if there is any evidence of model overfit or underfit.
    - Chose an appropriate data augmentation strategy to resolve underfitting/overfitting 
    - Model Building & training on the augmented data :
        - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale - images to normalize pixel values between (0,1).
        - Choose an appropriate optimiser and loss function for model training
        - Train the model for ~20 epochs
        - Write your findings after the model fit, see if the earlier issue is resolved or not?
    - Class distribution: Examine the current class distribution in the training dataset 
        - - Which class has the least number of samples?
        - - Which classes dominate the data in terms of the proportionate number of samples?
    - Handling class imbalances: Rectify class imbalances present in the training dataset with Augmentor library.
    - Model Building & training on the rectified class imbalance data :
        -Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale - images to normalize pixel values between (0,1).
        - Choose an appropriate optimiser and loss function for model training
        - Train the model for ~30 epochs
        - Write your findings after the model fit, see if the issues are resolved or not?
    > Dataset
    - The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

    The data set contains the following diseases:

    ![datasetdf](https://github.com/kshitij-raj/Melanoma-Skin-Cancer-Detection/blob/f143d178495ec6490ce2ee18c4cbbfb2e1388cea/Readme_images/Datasetdf.png)

    ![datasetplot](https://github.com/kshitij-raj/Melanoma-Skin-Cancer-Detection/blob/f143d178495ec6490ce2ee18c4cbbfb2e1388cea/Readme_images/DatasetPlot.png)

    To overcome the issue of class imbalance, used a python package  Augmentor (https://augmentor.readthedocs.io/en/master/) to add more samples across all classes so that none of the classes have very few samples.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

### Sample image from Dataset

![sample image](https://github.com/kshitij-raj/Melanoma-Skin-Cancer-Detection/blob/b43daf05e84626d3796321e79caeb2f6f8179346/Readme_images/Samleimagefromdataset.png)

## CNN Architecture Design
To classify skin cancer using skin lesions images. To achieve higher accuracy and results on the classification task, I have built custom CNN model.

- Rescalling Layer - To rescale an input in the [0, 255] range to be in the [0, 1] range.
- Convolutional Layer - Convolutional layers apply a convolution operation to the input, passing the result to the next layer. A convolution converts all the pixels in its receptive field into a single value. For example, if you would apply a convolution to an image, you will be decreasing the image size as well as bringing all the information in the field together into a single pixel. 
- Pooling Layer - Pooling layers are used to reduce the dimensions of the feature maps. Thus, it reduces the number of parameters to learn and the amount of computation performed in the network. The pooling layer summarises the features present in a region of the feature map generated by a convolution layer.
- Dropout Layer - The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
- Flatten Layer - Flattening is converting the data into a 1-dimensional array for inputting it to the next layer. We flatten the output of the convolutional layers to create a single long feature vector. And it is connected to the final classification model, which is called a fully-connected layer.
- Dense Layer - The dense layer is a neural network layer that is connected deeply, which means each neuron in the dense layer receives input from all neurons of its previous layer.
- Activation Function(ReLU) - The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.The rectified linear activation function overcomes the vanishing gradient problem, allowing models to learn faster and perform better.
- Activation Function(Softmax) - The softmax function is used as the activation function in the output layer of neural network models that predict a multinomial probability distribution. The main advantage of using Softmax is the output probabilities range. The range will 0 to 1, and the sum of all the probabilities will be equal to one.

### Model Architecture
![Model Arch](https://github.com/kshitij-raj/Melanoma-Skin-Cancer-Detection/blob/d8b2ca8cc296af14ab9aa7a6def31a7efc86271b/Readme_images/ModelLayer.png)

### Model Evaluation
![ModelEvaluation](https://github.com/kshitij-raj/Melanoma-Skin-Cancer-Detection/blob/7e7a17d3c891bf12be42385979168135775654c4/Readme_images/ModelEvaluation.png)

## Conclusions
- optimizer="Adam",loss="categorical_crossentropy" are used.
- To overcome the issue of class imbalance, used a python package Augmentor (https://augmentor.readthedocs.io/en/master/) to add more samples across all classes so that none of the classes have very few samples.
- To prevent overfit we have used early stopping when val_accuracy had stopped improving
- Finding on the first base model is that the model is overfitting because we can also see difference in loss functions in training & test around the 10-11th epoch
- The accuracy is just around 75-80% because there are enough features to remember the pattern.
- But again, it's too early to comment on the overfitting & underfitting debate
- Finding from Custom Model - There is no improvement in accuracy but we can definitely see the overfitting problem has solved due to data augmentation and We can increase the epochs to increase the accuracy so it's too early for judgement

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used


- glob - version 1.0
- pathlib - version 1.0
- tensorflow - version 1.0
- matplotlib - version 1.0
- numpy - version 1.0
- pandas - version 1.0
- os - version 1.0
- PIL - version 1.0
- tensorflow - version 1.0
- sklearn - version 1.0
- tqdm - version 1.0
- collections - version 1.0
- IPython - version 1.0
- time - version 1.0
- matplotlib - version 1.0
- seaborn - version 1.0
<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
Give credit here.
- This project was inspired by Upgrad.
- This project was based on Upgrad's Tutorial.


## Contact
Created by [@avs-abhishek123] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
