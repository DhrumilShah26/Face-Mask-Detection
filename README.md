# Face-Mask-Detection

The model gets the image with the face and run it through a cascade classifier. The classifier will give the region of interest of the face (height and width). Secondly, the model will resize the region of interest into a 100x100 and pass it to a pre-trained CNN, it will give us the probability as an output.

Step 1: Data Preprocessing:
The dataset consists of images with different colors, different sizes, and different orientations. Therefore, it needs to convert all the images into grayscale because color should not be a critical point for detecting mask. After that, the images must be in the same size (100x100) before applying it to the neural network.

Step 2: Training the CNN:
This consists of 2 convolutional layers (Two Convo2D 100@3x3). First, you have to load the dataset from data preprocessing. Then you have to configure the convolutional architecture. I’ve included a model.add(Dropout(0.5)) to get rid of overfitting. Since we have two categories(with mask and without mask) we can use binary_crossentropy. You start training for 20 epoch with a model checkpoint.

Step 3: Detecting Faces with and without Masks:
First, you have to load the model that we created. Then we set the camera we want as the default.

Secondly, we need to label the two probabilities (0 for with_mask and 1 for without_mask). After that, we need to set the bounding rectangle color using RGB values. I’ve given RED and GREEN as two colors.
Inside an infinite loop, we are going to read frame by frame from the camera and convert them to grayscale and detect the faces. And it will be run through a for loop to for each face and detect the region of interest, resize and reshape it to 4D since the training network expects 4D input. For the model, we are going to use the best model available to get the result. This result consists of the probability (result=[P1, P2]) of the with a mask or without a mask. It will be labeled after that.
