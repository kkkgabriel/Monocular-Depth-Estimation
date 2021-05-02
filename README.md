
# Monocular Depth Estimation
Estimating depth is an essential task in the field of robotics, it is an important aspect of projects such as autonomous vehicles. Typically, getting depth from a scene in real time is tasked to a LiDAR or a depth sensing camera. Nevertheless, these are expensive equipment and not feasible to implement in small-scale projects. Thus, estimating depth from monocular images emerged as a field of study, using deep learning to map the image from a scene to its depth map.<br>
As humans, we perceive depth with our binocular vision (with two eyes). However, even with monocular vision (with one eye), we can also perceive depth based on some monocular cues . For example:
*	Relative size: Nearer objects would appear larger in the image
*	Texture gradient: Textures of nearer objects would appear more defined while distant objects would seem unclear
*	Interposition: Nearer objects would block the vision of more distant objects
*	Shadow: Shadows of distant objects may seem smaller
*	Lighting: Distant objects may seem darker.<br>
This project aims to develop a model that is able to map an image to its depth map. And understand what are the cues that the model picks up on in order to do so.<br>

# Dataset
We are using the NYUv2 dataset, which can be downloaded [here](https://www.googleapis.com/drive/v3/files/10UaPxzVFyepzXfEf-ODsgJ_0n8srZatm?alt=media&key=AIzaSyCkMUZY02iddrkxpM32Cb6_2nR8oWaBMw8).
More information can be found on the [NYUv2 dataset site](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

# Instructions to run code
1. ```pip install requirements.txt``` to install all the required python libraries. We advise you to do so in a venv. Our version ran on ```python 3.9.4```. Its best if you do the same. Otherwise, you may have to verify the torch installations.
2. Run codes
- ```python train.py ``` trains a new model with our architecture.
- ```python test.py``` calculates the scores of a model based on some of our metrics. 
- ```python predict.py``` uses a model to predict depth of a selected image from the testset
- ```depth_estimationv2.ipynb``` contains the entire workflow of our model, including training, testing, interpretability and attacks on the model.

# Structure
- ```models/``` stores the models that we have trained, you can store your new model there.
- ```utils/``` stores the python classes and methods that we call in the codes.
-```data/``` should store the ```nyu_data.zip``` that you have to download (See Dataset section).

# Credits
Nathan Silberman and NYU team
