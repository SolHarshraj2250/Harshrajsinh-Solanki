# 1.) Project Description:-
# Project Title:- Multiple Type of Vehicle Counting From Video
## Description:-
Currently We all know Traffic and Road Maintaince is one of the biggest issue across all over the World. So overcome that Problem We have tried to implement this Project.In this Project, I had used Python as my base lanquage and I had used Pytorch and Pyqt5 kind of Apis and other useful Apis to complete this project with best of my Effort.
Also We have taken the help of the yolov3 and sort algorithm for object detection to implement this project.
# APIS Information (Which New API We have Used and For What Purpose):-
- torch:-
Torch is an open-source machine learning library, a scientific computing framework, and a script language based on the Lua programming language. 
It provides a wide range of algorithms for deep learning, and uses the scripting language LuaJIT, and an underlying C implementation.

- torch.nn.functional:-
functional provides some layers / activations in form of functions that can be directly called on the input rather than
defining the an object. For example, in order to rescale an image tensor, you call torch

- glob:-
Glob is a general term used to define techniques to match specified patterns according to rules related to Unix shell.

- PIL:-
Python Imaging Library. Python Imaging Library (abbreviated as PIL) (in newer versions known as Pillow) is a free library 
for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.

- torch.utils.data:-
It automatically converts NumPy arrays and Python numerical values into PyTorch Tensors. It preserves the data structure,
 e.g., if each sample is a dictionary, it outputs a dictionary with the same set of keys but batched Tensors as values 
(or lists if the values can not be converted into Tensors).

- torchvision.transforms:-

- tenserflow:-
It is an open source artificial intelligence library, using data flow graphs to build models.
It allows developers to create large-scale neural networks with many layers. TensorFlow is mainly used for:
Classification, Perception, Understanding, Discovering, Prediction and Creation.

-  __future__-dividion,print_function:-
The future statement is intended to ease migration to future versions of Python that introduce incompatible changes
to the language. It allows use of the new features on a per-module basis before the release in which the feature becomes standard. and then access it as usual.

- tqdm:-
TQDM is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.

- torch.autograd:-
Autograd is now a core torch package for automatic differentiation. It uses a tape based system for automatic differentiation.
In the forward phase, the autograd tape will remember all the operations it executed, and in the backward phase, it will replay the operations.

- matplotlib.patches:-
- skimage-io:-
- argparse:-
- filterpy.kalman:-
- cv2:-
- PyQt5.Qtcore:-
- Qtgui,QtWidgets:-
- copy:-

# 2.) Project Running:-
- Windows/Linux
- python>=3.5

## Steps For Runnning this Project:-
- Install Pytorch and PyQt5 in your Anaconda Navigator From Anaconda Prompt through Below Commands
- This yolov3 file is big. So i have posted link overhere So before moving further step you need to download this in your pc and you have to paste it into "Algo" File which i have Submitted Empty for that.
The Link is:-[link_(Dataset)](https://pjreddie.com/media/files/yolov3.weights)
- After that please run this project's main file through "Anaconda Command Prompt" by giving path of that specific file. And then write:- python Main.py
- After that GUI will open and First You need to Select the video From "input-Output" File (Which i have puted overhere) and then Select the appropriate Area by clicking double click on mouse and then Start the video then You will be able to see the Proper Live Running of Project.
- After that Final Result will be stored in final.txt in Final directory with Proper Format.
Like the Format is:- [videoname, id, objectname] for each Vehicle. For Example:- demo1 1 car etc.

# 3.) Sample Input-Output:-
![Output11](Input-Output/Output1.mp4)

![Output2](Input-Output/Output2.mp4)

# 4.) Challenges:-
This Project Can Run For Most of the Real World Videos but it depends on the speed of the video.


