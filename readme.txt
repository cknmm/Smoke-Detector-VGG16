This project was created as a submission to HACK-SRM 4.0.

This project is an application of deep learning where a smoke detector is created that works on footage. It can detect the bounding box of a smoke that appears in a frame of a video. Detection of such bounding boxes is known as bounding box regression.

NOTE: The model.h5 must be downloaded from "https://drive.google.com/file/d/1hmwDPn1VIOayiLQRA2MtwVxvidY7tDIZ/view?usp=drivesdk" and placed in the same directory as the code. The dataset must also be downloaded from "https://drive.google.com/file/d/1hnx637fF8tJRRKGTyx2TYuOH_MiRo14r/view?usp=drivesdk" and unzipped in the same folder (The path to this dataset can be specified in the visual_tool.py file)

Bounding box regression was achieved with the help of numerous technologies. A very deep convolutional neural network (CNN) architecture - VGG16 - was used for the base model with each convolutional layer followed by the RELU activation function. The initialisation weights of VGG16 were taken from a pre-trained network trained on the famous image-net classification dataset to perform transfer learning and get good initial weights as the starting point. The CNN was then followed by some Fully Connected Neural Network layers (FCNN) to get the final regression value. 

The model was trained for 16 epochs and each epoch took around 3 minutes on average. Mean Absolute Error was used as the loss function and Adam with modified parameters was used as the optimizer.

The output of the neural network is a set of 4 numbers that represent a rectangle where the smoke is present. The output can be represented as [xmin, ymin, xmax, ymax] coordinates of the rectangle. It is visualised using opencv in the visual_tool.py file where green rectangle is actual bounding box and the blue rectangle is the predicted bounding box. The evaluation accuracy of the model was around 96% and the average evaluation loss was observed to be 16.