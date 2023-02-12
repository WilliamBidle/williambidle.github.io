---
layout: post
title:  "Demystifying Machine Learning"
date:   2023-02-07 12:03:00 -0500
category: programming-projects
author: William Bidle
image: \assets\images\posts\Dymistifying Machine Learning.png
comments:  true
tags: machine-learning artificial-intelligence math neural-networks
---

As artificial intelligence has gained more attention in the media recently with the recent releases of DALL-E and ChatGPT, more and more people seem to become entranced with the power of machine learning. As someone with a heavy background in physics and mathematics, I find it really amazing how all of these models mostly boil down to some clever linear algebra and calculus, barring the details about the data they're trained on (for an interesting example of how training data can affect performance in machine learning, see my article on <a href="https://williambidle.github.io/programming-projects/2022/12/15/Fun-with-Autoencoders.html" target = "_blank">autoencoders</a>). So I figured it would be an interesting exercise to dive into the mathematics of how machines can actually learn, primarily through neural networks, with the hopes of dymstifying machine learning for those inside and outside of the field. This article serves as a companion piece to a joint project I worked on with <a href="https://ilanazane.github.io/" target = "_blank">Ilana Zane</a> called <a href="https://williambidle.github.io/programming-projects/2023/02/08/Baby-TensorFlow.html" title="Baby TensorFlow" target = "_blank">Baby Tensorflow</a>.

# **Setting the Stage**

In very simplistic terms, the main point of a neural network is that given some input data, can it accurately predict what the corresponding output for that input should be? A common example of this is image classication, such as whether an image is of a cat or a dog. As humans, we can do this with ease, so it seems like a pretty silly thing to train a machine to do. What we don't realize, however, is that we have been trained since birth to recognize patterns in almost every aspect of life, and it is first knowledge to us that cats may have more complex patterns on their fur, or that dogs tend to hang longer faces. With neural networks, they can be trained to accurately identify the two different species within only seconds of training. In many cases, they have the ability to pick out patterns in things we could never hope to even with the most keen eyes, and this is only one of the many areas in which machine learning is useful (the list of use cases nowadays seems nearly endless).

In many ways this seems like magic, and without any insight into how these models work, it's a fairly understandable viewpoint. At the end of the day, however, all these models are taught to do is minimize the chance that their output is incorrect, and then give you the most probable results. There is really no 'thinking' going on at all on the machine's end at all, against what many people may think. In very simplistic terms, models such as ChatGPT have essentially been trained to give you response text based on likilhood, and while it's still really impressive, it's just math! What I would like to do in this article is to peel back the layers and brute force through the mathematics behind how a simple neural network works, and extending the discussion to larger, more complex networks. The mathematics will involve some basic linear algebra and calculus, but even if you are uncomfortable with these things, there still should be a lot for you to take away (see the bottom of this article for some additional resources).

# **A Simple Neural Network**

Since the notation can get a little cumbersome and hard to follow for large neural networks, I am going to stick with an easy one containing 3 total layers, with 3 nodes in the first layer (the input layer), denoted $$\vec{x}^{(1)}$$, 2 nodes in the second layer (the hidden layer), denoted $$\vec{x}^{(2)}$$, and 2 nodes for the final layer (the output layer), denoted $$\vec{x}^{(3)}$$. Each layer will be densely connected by a set of weights, $$\underline{w}^{(1)}$$ and $$\underline{w}^{(2)}$$. Each input will also have a corresponding label, $$\vec{y}$$, and we can express each of these quantities compactly in vector and matrix notation as follows:

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

$$ \vec{y} = \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} $$

We will get to $$\vec{x}^{(2)}$$ and $$\vec{x}^{(3)}$$ in a moment since they can actually be expressed solely in terms of the above quantities. Now, since the most important part of machine learning is the loss function, we also need to define that. As mentioned previously, the loss function is a metric for evaluating the neural network's performance by comparing the output layer, in our case $$ \vec{x}^{(3)}$$, to the actual result, in our case $$ \vec{y}$$. For this example, we will use Mean Squared Error (MSE), which can be expressed by:

$$ MSE = \sum_{i} (x_i^{(3)} - y_i)^2 $$

Now that we have defined our known quantities, we can begin moving through the network to determine the values of the other layers. First, we will start with $$\vec{x}^{(2)}$$. Now we first must connect all of the input layers through their respective weights:

Now is where we introduce the activation function sigmoid:

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

Therefore $$ \vec{x}^{(2)} $$ is given by:

$$ \vec{x}^{(2)} = \sigma(\underline{w}^{(1)} \cdot \vec{x}^{(1)}) $$

or written out fully:

$$ \vec{x}^{(2)} = \begin{bmatrix} \sigma (w_{11}^{(1)}x_1 + w_{12}^{(1)}x_2 + w_{13}^{(1)}x_3) \\ \sigma (w_{21}^{(1)}x_1 + w_{22}^{(2)}x_2 + w_{23}^{(1)}x_3) \end{bmatrix} $$

Now, we will move on to $$ \vec{x}^{(3)} $$:

$$ \vec{x}^{(3)} = \sigma(\underline{w}^{(2)} \cdot \vec{x}^{(2)}) = \sigma(\underline{w}^{(2)} \cdot \sigma(\underline{w}^{(1)} \cdot \vec{x}^{(1)})) $$

or written out fully:

$$ \vec{x}^{(3)} = \begin{bmatrix} \sigma (w_{11}^{(2)}\sigma (w_{11}^{(1)}x_1 + w_{12}^{(1)}x_2 + w_{13}^{(1)}x_3) + w_{12}^{(2)}\sigma (w_{21}^{(1)}x_1 + w_{22}^{(2)}x_2 + w_{23}^{(1)}x_3)) \\ \sigma (w_{21}^{(2)}\sigma (w_{11}^{(1)}x_1 + w_{12}^{(1)}x_2 + w_{13}^{(1)}x_3) + w_{22}^{(2)}\sigma (w_{21}^{(1)}x_1 + w_{22}^{(2)}x_2 + w_{23}^{(1)}x_3)) \end{bmatrix} $$

Now plugging this expression into the MSE equation from above:

$$ MSE = \left[ \left( \sigma (w_{11}^{(2)}\sigma (w_{11}^{(1)}x_1 + w_{12}^{(1)}x_2 + w_{13}^{(1)}x_3) + w_{12}^{(2)}\sigma (w_{21}^{(1)}x_1 + w_{22}^{(2)}x_2 + w_{23}^{(1)}x_3)) - y_1 \right)^2 \\ + \left( \sigma (w_{21}^{(2)}\sigma (w_{11}^{(1)}x_1 + w_{12}^{(1)}x_2 + w_{13}^{(1)}x_3) + w_{22}^{(2)}\sigma (w_{21}^{(1)}x_1 + w_{22}^{(2)}x_2 + w_{23}^{(1)}x_3)) - y_2 \right)^2 \right] $$

This forms the basis for our simple neural network. Just by knowing the input values and their corresponding labels, we can compute the loss of the network. This serves as a measurement for how good the current weights are. 

# **How a Neural Network Learns**

Now we can get into the main part of this article, how a neural network can actually learn from the information it is given. The basic idea is that depending on how good the current weights are in the network, we need to nudge them in the right direction to minimize the network's loss. This methodology is called _backpropagation_, and is an essential concept in machine learning. If the loss of some given input is very high, then we must nudge the weights a little harder than if the loss is low. While this seems like a somewhat complicated procedure, it's actually extremely simple when we realize that this is just a simple optimization problem, and when we are talking optimization, we are talking calculus. 

In order to minimize/maximize a function, we need to look at the derivitive of that function with respect to its inputs. The derivitive of a function will tell us how that function will change as we change its inputs. Right away we can see how this might be useful to our problem, because our function of interest is the MSE loss function from above, and the weights make up the inputs we wish to change. 


Therefore, we wish to find the gradients of the MSE loss function with respect to the different weights:

$$
\nabla_{\underline{w}^{(1)}}MSE = 
\begin{bmatrix}
\frac{\partial MSE}{\partial w_{11}^{(1)}} & \frac{\partial MSE}{\partial w_{12}^{(1)}} & \frac{\partial MSE}{\partial w_{13}^{(1)}} \\
\frac{\partial MSE}{\partial w_{21}^{(1)}} & \frac{\partial MSE}{\partial w_{22}^{(1)}} & \frac{\partial MSE}{\partial w_{23}^{(1)}}
\end{bmatrix}
$$

$$
\nabla_{\underline{w}^{(2)}}MSE = 
\begin{bmatrix}
\frac{\partial MSE}{\partial w_{11}^{(2)}} & \frac{\partial MSE}{\partial w_{12}^{(2)}} \\
\frac{\partial MSE}{\partial w_{21}^{(2)}} & \frac{\partial MSE}{\partial w_{22}^{(2)}}
\end{bmatrix}
$$

Notice how these gradient matricies dimensionally match up with the definition of the weight matricies from above. In the compact notation, our algorithm is as follows:

$$
\underline{w}^{(1)}_{\; new} = \underline{w}^{(1)}_{\; old} - \epsilon \nabla_{\underline{w}^{(1)}}MSE
$$

$$
\underline{w}^{(2)}_{\; new} = \underline{w}^{(2)}_{\; old} - \epsilon \nabla_{\underline{w}^{(2)}}MSE
$$

where $$\epsilon$$ is called the _learning rate_, and sets how fast or slow we let the network learn. 

# **Diving into the Math**

More Coming Soon!

# **Conclusion**

I am hoping this article was informative and helpful to anyone interested in the topic. At the end of the day, a lot of the really complex AI bots we are seeing today just boil down to some clever mathematics, and if you're willing to dive into some of the details you will most certainly come out with a better appreciation of how they work. If you're interested in learning more, I highly recommend checking out <a href="https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown" target = "_blank">3Blue1Brown's YouTube series on machine learning</a>, as well as <a href="https://vas3k.com/blog/machine_learning/?ref=hn" title="Baby TensorFlow" target = "_blank">vas3k's</a> blog post about the different types of machine learning. 







<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>