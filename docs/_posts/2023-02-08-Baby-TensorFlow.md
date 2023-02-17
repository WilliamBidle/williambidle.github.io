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

This post summarizes a public machine learning project I worked on in collaboration with <a href="https://ilanazane.github.io/" target = "_blank">Ilana Zane</a>. In our search to better understand machine learning, we ended up building our machine learning framework modeled after <a href="https://www.tensorflow.org/" target = "_blank">TensorFlow</a>, which we call _Baby Tensorflow_. All of the code and instructions on how to download the software can be found <a href="https://github.com/WilliamBidle/Baby-TensorFlow" target = "_blank">here</a>.

If you have ever had any experience with  machine learning in Python, its pretty likely you've heard of TensorFlow. In essence, TensorFlow is a free and open-source software library for machine learning in Python that modularizes the process into a simple, building block-like style. It allows for complex neural networks to be built in just a few lines, and is extremely user friendly. As a demonstration, let's build a model to perform image classification:

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

See how easy that is? You are able to add layers in a very simple and intuitive manner, and have a ton of control over the different hyperparameters, such as the number of nodes per layer, the activation functions to use, what metrics to track, and so much more. Practically every situation involving machine learning can utilize TensorFlow to perform the task at hand. Towards the end of the article we will revist TensorFlow and compare some working examples to ones done in our own architecture. 

Now I won't be going over any of the mathematics used in building our model, but for an discussion on how neural networks work in general, check out my post on <a href="/programming-projects/2023/02/07/Demystifying-Machine-Learning.html" title="Demystifying Machine Learning" target = "_blank">Demystifying Machine Learning</a>.

# Baby TensorFlow

More coming soon!

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>