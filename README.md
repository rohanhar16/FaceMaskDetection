  <body>
    <h1>Face Mask Detection using Deep Learning</h1>
    <p>This project is an implementation of a deep learning model for face mask detection, using the Keras framework. The model is trained on a dataset of images containing people with and without masks, and can predict whether a person is wearing a mask or not, given an input image.</p>
    <h2>Model Architecture</h2>
    <p>The model architecture is based on a convolutional neural network (CNN), with the following layers:</p>
    <ul>
      <li>Input layer: takes input images of size 150x150x3.</li>
      <li>Convolutional layer: 100 filters of size 3x3, with ReLU activation.</li>
      <li>Max pooling layer: pool size of 2x2.</li>
      <li>Convolutional layer: 100 filters of size 3x3, with ReLU activation.</li>
      <li>Max pooling layer: pool size of 2x2.</li>
      <li>Flatten layer: flattens the output of the previous layer.</li>
      <li>Dropout layer: randomly drops 50% of the neurons to prevent overfitting.</li>
      <li>Dense layer: 50 neurons with ReLU activation.</li>
      <li>Output layer: 2 neurons with softmax activation, representing the probabilities of the two classes (mask and no mask).</li>
    </ul>
    <p>The model is compiled with the following hyperparameters:</p>
    <ul>
      <li>Optimizer: Adam.</li>
      <li>Loss function: binary cross-entropy.</li>
      <li>Metrics: accuracy.</li>
    </ul>
    <h2>Dataset</h2>
    <p>The dataset used for training and validation consists of two classes: images with masks and images without masks. The dataset is split into a training set and a validation set, with a batch size of 10.</p>
    <p>The training images are preprocessed using various data augmentation techniques, including rotation, shifting, shearing, zooming, flipping, and filling.</p>
    <h2>Training</h2>
    <p>The model is trained for 10 epochs using the training set, with early stopping based on the validation loss. Model checkpoints are saved after each epoch.</p>
    <h2>Dependencies</h2>
    <ul>
      <li>cv2</li>
      <li>numpy</li>
      <li>keras</li>
    </ul>
    <h2>Code Explanation</h2>
    <ul>
        <li>We load the trained Keras model using <code>load_model</code> method and <code>model-010.h5</code> file.</li>
        <li>We define <code>labels_dict</code> and <code>color_dict</code> which maps label values to human-readable strings and color codes.</li>
        <li>We set the size parameter to 4 and initialize the webcam object.</li>
        <li>We load the pre-trained face detection model from <code>haarcascade_frontalface_default.xml</code> file.</li>
        <li>We use a while loop to capture frames from the webcam and detect faces in the video stream.</li>
        <li>For each face detected, we extract the face region and resize it to 150x150.</li>
        <li>We normalize the image data by dividing it with 255.</li>
        <li>We reshape the image to a 4D tensor and apply <code>predict</code> method of the Keras model to classify whether the person is wearing a mask or not.</li>
        <li>We draw a rectangle around the face region and display the result on the screen.</li>
    </ul>
    <h2>How to use</h2>
    <ol>
        <li>Make sure you have all the dependencies installed.</li>
        <li>Download <code>model-010.h5</code> and <code>haarcascade_frontalface_default.xml</code> files.</li>
        <li>Run the code. It will launch the webcam and start detecting faces in real-time.</li>
        <li>Press ESC key to exit the program.</li>
    </ol>
    <h2>Results</h2>
    <p>After training, the model achieves an accuracy of 94.03% on the validation set.</p>
