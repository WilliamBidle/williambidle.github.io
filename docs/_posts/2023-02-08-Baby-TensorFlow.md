---
layout: post
title:  "Baby TensorFlow"
date:   2023-02-08 11:13:00 -0500
category: programming-projects
author: William Bidle
comments: true
tags: artificial-intelligence machine-learning autoencoders python
image: \assets\images\posts\Baby TensorFlow.png

---

If you have ever had any experience with machine learning in Python its pretty likely you've heard of <a href="https://www.tensorflow.org/" target = "_blank">TensorFlow</a>. In essence, TensorFlow is a free and open-source software library for machine learning in Python that modularizes the process into a simple, building block-like style. It allows for complex neural networks to be built in just a few lines:

```python
# Importing the required Keras modules containing model and layers
import tensorflow as tf

# Building the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='sigmoid'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model and define important model parameters
model.compile(optimizer='SGD',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

```

See how easy that is? You are able to add layers in a very simple and intuitive manner, and have a ton of control over the different hyperparameters, such as the number of layers and nodes per layer. Additionally, you can ...

 It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks.

I won't be going over any of the mathematics used in building the model, but for an discussion on how neural networks such as this work, see my post on <a href="/programming-projects/2023/01/12/Demystifying-Machine-Learning.html" title="Demystifying Machine Learning" target = "_blank">Demystifying Machine Learning</a>. This project was done in collboration with <a href="https://ilanazane.github.io/" target = "_blank">Ilana Zane</a>, and the code for the project can be found on my <a href="https://github.com/WilliamBidle/Baby-TensorFlow" target = "_blank">Github</a>.

# Baby TensorFlow

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>