---
layout: post
title:  "Baby TensorFlow"
date:   2023-01-09 12:13:00 -0500
category: programming-projects
author: William Bidle
comments: true
tags: artificial-intelligence machine-learning autoencoders python
image:

---

If you have ever had any experience with machine learning in Python its pretty likely you've heard of <a href="https://www.tensorflow.org/" target = "_blank">TensorFlow</a>. In essence, TensorFlow is a free and open-source software library for machine learning in Python that modularizes the process into a simple, building block-like style. It allows for complex neural networks to be built in just a few lines:

```python
# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
```

See how easy that is? You are able to add layers in a very simple and intuitive manner, and have a ton of control over the different hyperparameters, such as the number of layers and nodes per layer. 

 It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks.

This project was done in collboration with <a href="https://ilanazane.github.io/" target = "_blank">Ilana Zane</a>, and the code for the project can be found on my <a href="https://github.com/WilliamBidle/Baby-TensorFlow" target = "_blank">Github</a>.

# Baby TensorFlow

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>