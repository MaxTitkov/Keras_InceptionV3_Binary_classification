# Keras_InceptionV3_Binary_classification

Keras implementation of InceptionV3 convolutional neural network in solving binary classification problems

The folder should be like this:

data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
Ordering:

1. bottleneck_features.py: Fetching the bottleneck features from pretrained InceptionV3 as numpy arrays;
2. top_model.py: Training FFN model with numpy arrays;
3. complete_model.py: Training the complete model with images;
4. visualization.py: Visualize the result.

