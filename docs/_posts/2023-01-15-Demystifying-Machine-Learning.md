---
layout: post
title:  "Demystifying Machine Learning"
date:   2023-01-12 12:03:00 -0500
category: programming-projects
author: William Bidle
image: \assets\images\posts\Dymistifying Machine Learning.png
comments:  true
tags: 
---

As artificial intelligence has gained more attention in the media recently with the recent releases of DALL-E and ChatGPT, more and more people seem to become entranced with the power of machine learning. As someone with a heavy background in physics and mathematics, I find it really amazing how all of these models really just boil down to some clever linear algebra and calculus (barring the details about the data they're trained on). So I figured it would be an interesting exercise to dive into the mathematics of how neural networks actually learn, with the hopes of dymstifying machine learning. This article also serves as a companion piece to a joint project I worked on with <a href="https://ilanazane.github.io/" target = "_blank">Ilana Zane</a> called <a href="/programming-projects/2023/01/09/Baby-TensorFlow.html" title="Baby TensorFlow" target = "_blank">Baby Tensorflow</a>.

# **A Simple Neural Network**

Since the notation can get a little cumbersome and hard to follow for larger and larger networks, I am going to stick with an easy one containing 3 total layers, with 3 nodes in the first layer (the input layer), 2 nodes in the second layer (the hidden layer), and 2 nodes for the final layer (the output layer). HAVE ILANA DRAW A PRETTY VERSION FOR ME!!!

$$ \vec{x}^{(1)} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} $$

$$
\underline{w}^{(1)} = 
\begin{bmatrix}
w_{11}^{(1)} & w_{12}^{(1)} & w_{13}^{(1)}\\
w_{21}^{(1)} & w_{22}^{(1)} & w_{23}^{(1)}
\end{bmatrix}
$$

$$
\underline{w}^{(2)} = 
\begin{bmatrix}
w_{11}^{(2)} & w_{12}^{(2)} \\
w_{21}^{(2)} & w_{22}^{(2)}
\end{bmatrix}
$$

$$ \vec{y} = \begin{bmatrix} y_1 \\ y_2 \\ x_3 \end{bmatrix} $$

Now that we have defined our known quantities, we can begin moving through the network to determine the values of the other layers. First, we will start with $$ \vec{x}^{(2)} $$. Now we first must connect all of the input layers through their respective weights:

Now is where we introduce the activation function sigmoid:

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

Therefore $$ \vec{x}^{(2)} $$ is given by:

$$ \vec{x}^{(2)} = \sigma(\underline{w}^{(1)} \cdot \vec{x}^{(1)}) $$

Now, we will move on to $$ \vec{x}^{(3)} $$:

$$ \vec{x}^{(3)} = \sigma(\underline{w}^{(2)} \cdot \vec{x}^{(2)}) = \sigma(\underline{w}^{(2)} \cdot \sigma(\underline{w}^{(1)} \cdot \vec{x}^{(1)})) $$

$$ MSE = \sum_{i} (x_i^{(3)} - y_i)^2 $$



# **How a Neural Network Learns**

Now we can get into the main part of this article, how a neural network can actually learn from the information it is given.

Backpropagation...

$$
\nabla_{\underline{w}^{(1)}} = 
\begin{bmatrix}
\frac{\partial}{\partial w_{11}^{(1)}} & \frac{\partial}{\partial w_{12}^{(1)}} & \frac{\partial}{\partial w_{13}^{(1)}} \\
\frac{\partial}{\partial w_{21}^{(1)}} & \frac{\partial}{\partial w_{22}^{(1)}} & \frac{\partial}{\partial w_{23}^{(1)}}
\end{bmatrix}
$$

$$
\nabla_{\underline{w}^{(2)}} = 
\begin{bmatrix}
\frac{\partial}{\partial w_{11}^{(2)}} & \frac{\partial}{\partial w_{12}^{(2)}} \\
\frac{\partial}{\partial w_{21}^{(2)}} & \frac{\partial}{\partial w_{22}^{(2)}}
\end{bmatrix}
$$

# **Conclusion**

I highly recommendchecking out <a href="https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown" target = "_blank">3Blue1Brown's YouTube series on machine learning</a> for a very well done visual representation of what this article was trying to convey. 




<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>