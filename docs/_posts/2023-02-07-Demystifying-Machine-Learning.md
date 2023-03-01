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

As artificial intelligence has gained more attention in the media recently with the recent releases of DALL-E and ChatGPT, more and more people seem to become entranced with the power of machine learning. As someone with a heavy background in physics and mathematics, I find it really amazing how all of these models mostly boil down to some clever linear algebra and calculus, barring the details about the data they're trained on (for an interesting example of how training data can affect performance in machine learning, see my article on <a href="https://williambidle.github.io/programming-projects/2022/12/15/Fun-with-Autoencoders.html" target = "_blank">autoencoders</a>). So I figured it would be an interesting exercise to dive into the mathematics of how machines can actually learn, primarily through neural networks, with the hopes of dymstifying machine learning for those inside and outside of the field. This article serves as a companion piece to a joint project I worked on with <a href="https://ilanazane.github.io/" target = "_blank">Ilana Zane</a> called <a href="/programming-projects/2023/02/08/Artifice.html" target = "_blank">Artifice</a>.

# **Setting the Stage (I Promise to Add Visuals Soon!)**

In very simplistic terms, the main point of a neural network is that given some input data, can it accurately predict what the corresponding output for that input should be? A common example of this is image classication, such as whether an image is of a cat or a dog. As humans, we can do this with ease, so it seems like a pretty silly thing to train a machine to do. What we don't realize, however, is that we have been trained since birth to recognize patterns in almost every aspect of life, and it is first knowledge to us that cats may have more complex patterns on their fur, or that dogs tend to hang longer faces. With neural networks, they can be trained to accurately identify the two different species within only seconds of training. In many cases, they have the ability to pick out patterns in things we could never hope to even with the most keen eyes, and this is only one of the many areas in which machine learning is useful (the list of use cases nowadays seems nearly endless).

In many ways this seems like magic, and without any insight into how these models work, it's a fairly understandable viewpoint. At the end of the day, however, all these models are taught to do is minimize the chance that their output is incorrect, and then give you the most probable results. There is really no 'thinking' going on at all on the machine's end at all, against what many people may think. In very simplistic terms, models such as ChatGPT have essentially been trained to give you response text based on likilhood, and while it's still really impressive, it's just math! What I would like to do in this article is to peel back the layers and brute force through the mathematics behind how a simple neural network works, and extending the discussion to larger, more complex networks. The mathematics will involve some basic linear algebra and calculus, but even if you are uncomfortable with these things, there still should be a lot for you to take away (see the bottom of this article for some additional resources).

# **A Simple Neural Network**

In general, we denote the different layers in a neural network as vectors (e.g., $$\vec{x}$$), with each element of the vector representing a node in that layer. Additionally, we denote the weights in a neural network as matricies (e.g., $$\underline{w}$$), with each column and row element representing a weight that connects a node from the previous layer to a node in the next layer. Since even this shorthand notation can get a little complicated and hard to follow for large neural networks, I am going to stick with an easy one containing 3 total layers, with 3 nodes in the first layer (the input layer), denoted $$\vec{x}^{(1)}$$, 2 nodes in the second layer (the hidden layer), denoted $$\vec{x}^{(2)}$$, and 2 nodes for the final layer (the output layer), denoted $$\vec{x}^{(3)}$$. Each layer will be densely connected by a set of weights, $$\underline{w}^{(1\rightarrow 2)}$$ and $$\underline{w}^{(2 \rightarrow 3)}$$, or in otherwords, each node from each layer is connected to each node in adjacent layers. Each input will also have a corresponding label, $$\vec{y}$$, and we can express each of these quantities compactly in vector and matrix notation as follows:

$$ \vec{x}^{(1)} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} $$

$$
\underline{w}^{(1 \rightarrow 2)} = 
\begin{bmatrix}
w_{11}^{(1 \rightarrow 2)} & w_{12}^{(1 \rightarrow 2)} & w_{13}^{(1 \rightarrow 2)}\\
w_{21}^{(1 \rightarrow 2)} & w_{22}^{(1 \rightarrow 2)} & w_{23}^{(1 \rightarrow 2)}
\end{bmatrix}
$$

$$
\underline{w}^{(2 \rightarrow 3)} = 
\begin{bmatrix}
w_{11}^{(2 \rightarrow 3)} & w_{12}^{(2 \rightarrow 3)} \\
w_{21}^{(2 \rightarrow 3)} & w_{22}^{(2 \rightarrow 3)}
\end{bmatrix}
$$

$$ \vec{y} = \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} $$

We will get to $$\vec{x}^{(2)}$$ and $$\vec{x}^{(3)}$$ in a moment since they can actually be expressed solely in terms of the above quantities. As we can see, $$\vec{x}^{(1)}$$ contains 3 elements, representing the 3 nodes in the input layer. As for the weights, the first subscript tells us which node this weight connects us to, and the second subscript tells us which node we can from in the previous layer. For example, $$ w_{23}^{(1 \rightarrow 2)} $$ connects the 3rd node from the first layer to the 2nd node in the 2nd layer. Now, since the most important part of machine learning is the loss function, we also need to define that. As mentioned previously, the loss function is a metric for evaluating the neural network's performance by comparing the output layer, in our case $$ \vec{x}^{(3)}$$, to the actual result, in our case $$ \vec{y}$$. For simplicity, let's use Mean Squared Error (MSE), which can be expressed by:

$$ MSE = \sum_{i} (x_i^{(3)} - y_i)^2 $$

where we are summing over all of the elements of our vectors. Now that we have defined our known quantities, we can begin moving through the network to determine the values of the other layers. First, we will start with $$\vec{x}^{(2)}$$. Now we first must connect all of the input layers through their respective weights. This can be done in a very compact way by using matrix multiplication:

$$ \vec{x}^{(2)} = \begin{bmatrix} x_1^{(2)} \\ x_2^{(2)} \end{bmatrix} = \underline{w}^{(1 \rightarrow 2)} \cdot \vec{x}^{(1)} $$


But for the sake of being complete, let's write it out fully:

$$ \vec{x}^{(2)} = \begin{bmatrix} w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3 \\ w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3 \end{bmatrix} $$

Now in principle this is all we need for this layer, however it is common practice in machine learning to utilize something called an _activation function_, which serves to better learn complex patterns within the data. All this amounts to doing is wrapping the result from above in some function. For our example, let's use the sigmoid function:

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

which essentially serves to confine the results of the output layer between 0 and 1, something which is very useful in many different machine learning applications, such as binary classification (e.g., the cat and dog classification discussed above). There are many other different types of activation functions such as _ReLU_ and _SoftMax_, but I won't be going into those here. After wrapping the output of $$ \vec{x}^{(2)} $$ in our sigmoid function we find that:

$$ \vec{x}^{(2)} = \begin{bmatrix} \sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) \\ \sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3) \end{bmatrix} $$

Now, we can move on to $$ \vec{x}^{(3)} $$, which is an idential process:

$$ \vec{x}^{(3)} = \begin{bmatrix} x_1^{(3)} \\ x_2^{(3)} \end{bmatrix} = \sigma(\underline{w}^{(2 \rightarrow 3)} \cdot \vec{x}^{(2)}) = \sigma(\underline{w}^{(2 \rightarrow 3)} \cdot \sigma(\underline{w}^{(1 \rightarrow 2)} \cdot \vec{x}^{(1)})) $$

or written out fully:

$$ \vec{x}^{(3)} = \begin{bmatrix} \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) \\ \sigma (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) \end{bmatrix} $$

and there we have it! We have a way of determining what the output of our neural network is given some input data. Now, plugging this expression into the MSE equation from above:

$$ MSE = \left[ \left( \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_1 \right)^2 \\ + \left( \sigma (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_2 \right)^2 \right] $$

This forms the basis for our simple neural network. Just by knowing the input values and their corresponding labels, we can compute the loss of the network, which serves as a measurement for how good the current weights are. As we shall see in the next section this measurement will give us a methodology for better adjusting the weights to make the output of the model as close as possible to the expected label. 

# **How a Neural Network Learns**

Now we can get into the main part of this article, how a neural network can actually learn from the information it is given. The basic idea is that depending on how good the current weights are in the network, we need to nudge them in the right direction to minimize the network's loss. This methodology is called _backpropagation_, and is an essential concept in machine learning. If the loss of some given input is very high, then we must nudge the weights a little harder than if the loss is low. While this seems like a somewhat complicated procedure, it's actually extremely simple when we realize that this is just a simple optimization problem, and when we are talking optimization, we are talking calculus. 

In order to minimize/maximize a function, we need to look at the derivitive of that function with respect to its inputs. The derivitive of a function will tell us how that function will change as we change its inputs. Right away we can see how this might be useful to our problem, because our function of interest is the MSE loss function from above, and the weights make up the inputs we wish to change. Therefore, we wish to find the gradients of the MSE loss function with respect to the different weights:

$$
\nabla_{\underline{w}^{(1)}}MSE = 
\begin{bmatrix}
\frac{\partial MSE}{\partial w_{11}^{(1 \rightarrow 2)}} & \frac{\partial MSE}{\partial w_{12}^{(1 \rightarrow 2)}} & \frac{\partial MSE}{\partial w_{13}^{(1 \rightarrow 2)}} \\
\frac{\partial MSE}{\partial w_{21}^{(1 \rightarrow 2)}} & \frac{\partial MSE}{\partial w_{22}^{(1 \rightarrow 2)}} & \frac{\partial MSE}{\partial w_{23}^{(1 \rightarrow 2)}}
\end{bmatrix}
$$

$$
\nabla_{\underline{w}^{(2)}}MSE = 
\begin{bmatrix}
\frac{\partial MSE}{\partial w_{11}^{(2 \rightarrow 3)}} & \frac{\partial MSE}{\partial w_{12}^{(2 \rightarrow 3)}} \\
\frac{\partial MSE}{\partial w_{21}^{(2 \rightarrow 3)}} & \frac{\partial MSE}{\partial w_{22}^{(2 \rightarrow 3)}}
\end{bmatrix}
$$

Notice how these gradient matricies dimensionally match up with the definition of the weight matricies from above. Once we are able to determine these matricies, we then have a way to update the old weights of our network by nudging them in the direction of steepest descent. In our compact notation, the algorithm for adjusting the old weights is as follows:

$$
\underline{w}^{(1 \rightarrow 2)}_{\; new} = \underline{w}^{(1 \rightarrow 2)}_{\; old} - \epsilon \nabla_{\underline{w}^{(1 \rightarrow 2)}}MSE
$$

$$
\underline{w}^{(2 \rightarrow 3)}_{\; new} = \underline{w}^{(2 \rightarrow 3)}_{\; old} - \epsilon \nabla_{\underline{w}^{(2 \rightarrow 3)}}MSE
$$

where I have defined the _learning rate_, $$\epsilon$$, which sets how fast or slow we let the network learn. In the next section, we will dive into calculating these derivitives, and we will find that within the mess of variables, an elegant and compact algorithm arises. 

# **Diving into the Math (Simple Version)**

Now let's begin by finding the simplier of the two gradients from above, the second layer weight gradient. As we shall see, these terms are less complex since they are further towards the end of the network. Let's first just look at the first matrix element. In order to compute this derivitive we need to use the chain rule:

$$
\frac{\partial MSE}{\partial w_{11}^{(2 \rightarrow 3)}} = \frac{\partial}{\partial w_{11}^{(2 \rightarrow 3)}} \left[ \left( \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_1 \right)^2 

\\ + \left( \sigma (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_2 \right)^2 \right] 

$$

$$
\\ =  2 * \left[ \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_1 \right ]
\\ * \left[ \frac{\partial}{\partial w_{11}^{(2 \rightarrow 3)}} \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - \frac{\partial}{\partial w_{11}^{(2 \rightarrow 3)}} y_1 \right ]
$$

$$
\\ =  2  * \left[ \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_1 \right ]
\\ * \sigma^{'} (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) \\ * \left[ \frac{\partial}{\partial w_{11}^{(2 \rightarrow 3)}} (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3)) + \frac{\partial}{\partial w_{11}^{(2 \rightarrow 3)}} (w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) \right]

$$

$$

\\ \frac{\partial MSE}{\partial w_{11}^{(2 \rightarrow 3)}} =  2 * \left[ \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_1 \right ]
\\ * \sigma^{'} (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) *\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3)
$$

Yikes! As gross as this might initially look, let's subsitute back in our previously defined quantities from above: 

$$
\frac{\partial MSE}{\partial w_{11}^{(2)}} =  2 * (x_1^{(3)} - y_1)*x_1^{(3) \;'}*x_1^{(2)}
$$

Which is much better looking. Quickly jumping to the results of the other three elements:

$$
\frac{\partial MSE}{\partial w_{12}^{(2)}} = 2 * (x_1^{(3)} - y_1)*x_1^{(3) \;'}*x_2^{(2)}
$$

$$
\frac{\partial MSE}{\partial w_{21}^{(2)}} = 2 * (x_2^{(3)} - y_1)*x_2^{(3) \;'}*x_1^{(2)}
$$

$$
\frac{\partial MSE}{\partial w_{22}^{(2)}} = 2 * (x_2^{(3)} - y_1)*x_2^{(3) \;'}*x_2^{(2)}
$$

Therefore, our resulting gradient matrix is:

$$
\nabla_{\underline{w}^{(2)}}MSE = 
\begin{bmatrix}
2 * (x_1^{(3)} - y_1)*x_1^{(3) \;'}*x_1^{(2)} & 2 * (x_1^{(3)} - y_1)*x_1^{(3) \;'}*x_2^{(2)} \\
2 * (x_2^{(3)} - y_1)*x_2^{(3) \;'}*x_1^{(2)} & 2 * (x_2^{(3)} - y_1)*x_2^{(3) \;'}*x_2^{(2)}
\end{bmatrix}
$$

and it is here where we can recognize a pattern within the matrix elements, allowing us to rewrite the expression as products of individual matrix and vector elements:

$$
\nabla_{\underline{w}^{(2)}}MSE 
= 
\begin{bmatrix}
2 * (x_1^{(3)} - y_1) & 0 \\
0 & 2 * (x_2^{(3)} - y_1)
\end{bmatrix}
*
\begin{bmatrix}
x_1^{(3) \;'} \\
x_2^{(3) \;'}
\end{bmatrix}
*
\begin{bmatrix}
x_1^{(2)} & x_2^{(2)} 
\end{bmatrix}
$$

We can recognize that the very first matrix term is just the Jacobian of the loss function, the second term is just the derivitve of that connection's output layer, and the third term is that connections input layer. The main reason to do this is because it becomes really nice to implement programmatically, and is actually faster than doing each element one by one (something called _vectorization_). As long as we are able to compute each of these terms, we then have a way to update our second set of weights! 

# **Diving into the Math (Complex Version)**

Now it's time to figure out how to determine the gradient for the inner most weights. This time around we are going to run into a lot more terms since the first set of weights are nested further into our loss function. Let's again go through the math for the first term:

$$
\frac{\partial MSE}{\partial w_{11}^{(1 \rightarrow 2)}} = \frac{\partial}{\partial w_{11}^{(1 \rightarrow 2)}} \left[ \left( \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_1 \right)^2 

\\ + \left( \sigma (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_2 \right)^2 \right] 

$$

$$

\\ =  2 * \left[ \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_1 \right ]
\\ * \left[ \frac{\partial}{\partial w_{11}^{(1 \rightarrow 2)}} \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - \frac{\partial}{\partial w_{11}^{(1 \rightarrow 2)}} y_1 \right ]
\\
+ 2 * \left[ \sigma (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_1 \right ]
\\ * \left[ \frac{\partial}{\partial w_{11}^{(1 \rightarrow 2)}} \sigma (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - \frac{\partial}{\partial w_{11}^{(1 \rightarrow 2)}} y_2 \right ]
$$


$$
\\ =  2  * \left[ \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_1 \right ]
\\ * \sigma^{'} (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) \\ * \left[ \frac{\partial}{\partial w_{11}^{(1 \rightarrow 2)}} (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3)) + \frac{\partial}{\partial w_{11}^{(1 \rightarrow 2)}} (w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) \right]

\\ +  2  * \left[ \sigma (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_2 \right ]
\\ * \sigma^{'} (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) \\ * \left[ \frac{\partial}{\partial w_{11}^{(1 \rightarrow 2)}} (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3)) + \frac{\partial}{\partial w_{11}^{(1 \rightarrow 2)}} (w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) \right]


$$

$$
\\ \frac{\partial MSE}{\partial w_{11}^{(2 \rightarrow 3)}} =  2  * \left[ \sigma (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_1 \right ]
\\ * \left[ \sigma^{'} (w_{11}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{12}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) \right] \\ * \left[w_{11}^{(2 \rightarrow 3)}\sigma^{'} (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3)*x_1 \right]

\\ +  2  * \left[ \sigma (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) - y_2 \right ]
\\ * \left[ \sigma^{'} (w_{21}^{(2 \rightarrow 3)}\sigma (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3) + w_{22}^{(2 \rightarrow 3)}\sigma (w_{21}^{(1 \rightarrow 2)}x_1 + w_{22}^{(1 \rightarrow 2)}x_2 + w_{23}^{(1 \rightarrow 2)}x_3)) \right] \\ * \left[ (w_{21}^{(2 \rightarrow 3)}\sigma ^{'} (w_{11}^{(1 \rightarrow 2)}x_1 + w_{12}^{(1 \rightarrow 2)}x_2 + w_{13}^{(1 \rightarrow 2)}x_3)) * x_1 \right]
$$

Again, yikes. Let's do the same as before and subsitute back in our previously defined quantities from above: 

$$
\\ \frac{\partial MSE}{\partial w_{11}^{(2 \rightarrow 3)}} =  2  *  (x_1^{(3)} - y_1) * x_1^{(3) \;'} *  (w_{11}^{(2 \rightarrow 3)}*x_1^{(2) \;'})*x_1

\\ +  2  *  (x_2^{(3)} - y_2) * x_2^{(3) \;'} * (w_{21}^{(2 \rightarrow 3)}*x_1^{(2) \;'}) * x_1
$$

Which is much better looking. Quickly jumping to the results of the other five elements:

$$
\\ \frac{\partial MSE}{\partial w_{12}^{(2 \rightarrow 3)}} =  2  *  (x_1^{(3)} - y_1) * x_1^{(3) \;'} *  (w_{11}^{(2 \rightarrow 3)}*x_1^{(2) \;'})*x_2

\\ +  2  *  (x_2^{(3)} - y_2) * x_2^{(3) \;'} * (w_{21}^{(2 \rightarrow 3)}*x_1^{(2) \;'}) * x_2
$$

$$
\\ \frac{\partial MSE}{\partial w_{13}^{(2 \rightarrow 3)}} =  2  *  (x_1^{(3)} - y_1) * x_1^{(3) \;'} *  (w_{11}^{(2 \rightarrow 3)}*x_1^{(2) \;'})*x_3

\\ +  2  *  (x_2^{(3)} - y_2) * x_2^{(3) \;'} * (w_{21}^{(2 \rightarrow 3)}*x_1^{(2) \;'}) * x_3
$$

$$
\\ \frac{\partial MSE}{\partial w_{21}^{(2 \rightarrow 3)}} =  2  *  (x_1^{(3)} - y_1) * x_1^{(3) \;'} *  (w_{12}^{(2 \rightarrow 3)}*x_2^{(2) \;'})*x_1

\\ +  2  *  (x_2^{(3)} - y_2) * x_2^{(3) \;'} * (w_{22}^{(2 \rightarrow 3)}*x_2^{(2) \;'}) * x_1
$$

$$
\\ \frac{\partial MSE}{\partial w_{22}^{(2 \rightarrow 3)}} =  2  *  (x_1^{(3)} - y_1) * x_1^{(3) \;'} *  (w_{12}^{(2 \rightarrow 3)}*x_2^{(2) \;'})*x_2

\\ +  2  *  (x_2^{(3)} - y_2) * x_2^{(3) \;'} * (w_{22}^{(2 \rightarrow 3)}*x_2^{(2) \;'}) * x_2
$$

$$
\\ \frac{\partial MSE}{\partial w_{23}^{(2 \rightarrow 3)}} =  2  *  (x_1^{(3)} - y_1) * x_1^{(3) \;'} *  (w_{12}^{(2 \rightarrow 3)}*x_2^{(2) \;'})*x_3

\\ +  2  *  (x_2^{(3)} - y_2) * x_2^{(3) \;'} * (w_{22}^{(2 \rightarrow 3)}*x_2^{(2) \;'}) * x_3
$$

Therefore, our resulting gradient matrix is:

$$
\nabla_{\underline{w}^{(2)}}MSE = 
\begin{bmatrix}
2 * (x_1^{(3)} - y_1)*x_1^{(3) \;'}*x_1^{(2)} & 2 * (x_1^{(3)} - y_1)*x_1^{(3) \;'}*x_2^{(2)} \\
2 * (x_2^{(3)} - y_1)*x_2^{(3) \;'}*x_1^{(2)} & 2 * (x_2^{(3)} - y_1)*x_2^{(3) \;'}*x_2^{(2)}
\end{bmatrix}
$$

and it is here where we can recognize a pattern within the matrix elements, allowing us to rewrite the expression as products of individual matrix and vector elements:

$$
\nabla_{\underline{w}^{(2)}}MSE 
= 
\begin{bmatrix}
x_1^{(3) \;'} & 0 \\
0 & x_2^{(3) \;'}
\end{bmatrix}
*
\begin{bmatrix}
w_{11}^{(2 \rightarrow 3)} & w_{21}^{(2 \rightarrow 3)} \\
w_{12}^{(2 \rightarrow 3)} & w_{22}^{(2 \rightarrow 3)}
\end{bmatrix}
*
\begin{bmatrix}
2 * (x_1^{(3)} - y_1) & 0 \\
0 & 2 * (x_2^{(3)} - y_1)
\end{bmatrix}
*
\begin{bmatrix}
x_1^{(2) \;'} \\
x_2^{(2) \;'}
\end{bmatrix}
*
\begin{bmatrix}
x_1 & x_2 & x_3
\end{bmatrix}
$$










The Rest Coming Soon!

# **Conclusion**

I am hoping this article was informative and helpful to anyone interested in the topic. At the end of the day, a lot of the really complex AI bots we are seeing today just boil down to some clever mathematics, and if you're willing to dive into some of the details you will most certainly come out with a better appreciation of how they work. If you're interested in learning more, I highly recommend checking out <a href="https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown" target = "_blank">3Blue1Brown's YouTube series on machine learning</a>, as well as <a href="https://vas3k.com/blog/machine_learning/?ref=hn" target = "_blank">vas3k's</a> blog post about the different types of machine learning. 







<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>