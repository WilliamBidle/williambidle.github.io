---
layout: post
title:  "Quantum State Tomography through Maximum Likelihood Estimation"
date:   2023-02-01 12:00:00 -0500
category: programming-projects
author: William Bidle
comments: true
tags: quantum-information python maximum-likelihood-estimation gradient-descent quantum-state-tomography 
image: /assets/images/posts/QST Post/QST_Website_Post.png
---

 I recently just finished up my Master's Degree in Physics at Stony Brook University and wanted to summarize some of the research I had done during my time there on Quantum State Tomography (QST). I don't actually expect many people outside (or even inside for that matter) the field of physics to understand most of the technical details of QST, so the idea of this post is to make my work as accessible as possible to those who are interested in learning more. An in depth summary can be found in my <a href="/assets/Quantum_State_Tomography.pdf" target="_blank">Master's Thesis</a>, and all of the corresponding code for the project can be found on my <a href="https://github.com/WilliamBidle/Quantum-State-Estimation" target = "_blank">Github</a>.

# **What is all of this Quantum Nonsense?**

Before I begin even talking about QST, I think it might be a good idea to first discuss what we mean when we use the word quantum. The word quantum certainly gets thrown around a lot now adays, and in some cases  (see Antman)

In the field of Quantum Information Science, one of the most fundamental and complex ideas is that. 

Build up to why we want to image states

When we build the quantum internet, we will undoubtably need to diagnose/characterize certain processes to make sure they are working correctly. The problem though, is that accurately measuring the properties of quantum states turns out to be extremely difficult. This is where QST comes into play. 

# **Enter Quantum State Tomography?**

The basic idea behind QST is that for a set of measured statistical values of an identically prepared quantum state, we are able deduce a best possible guess for what that quantum state is. The idea is not too dissimilar from a CT scan of the brain, where several different 2D snapshots are layered together in order to create an accurate 3D rendering of what is actually going on.   

<center><figure><img src="/assets/images/posts/QST Post/Brain.png" style="width:50%;height:50%;"><figcaption>CT scan of a brain [<a href="https://en.wikipedia.org/wiki/Computed_tomography_of_the_head" target = "_blank">Image Source</a>]</figcaption></figure></center>


Now typically it is very easy to deduce what a quantum state will look like if you are given the state itself. However, the opposite situation is extremely difficult: If I give you what the quantum state looks like, can you tell me what the state is without any prior information? Going back to our simple metaphor of ...

# **Maximum Likelihood Estimation**

Now the process of using Maximum Likelihood Estimation (MLE) to predict a quantum state generally boils down to some complicated differential mathematics involving matrix algebra and quantum operators (see my thesis above for details), but to be honest, the details are irrelevant for demonstrating the idea. Going back to the analogy of imaging a complex object, pretend you were tasked with figuring out what the shape of some mystery 3D object was. How would you go about doing this? Since we can only see one side (or one 2D projection) of the shape a time, you might start by picking up the object and turning it to look at it from different angles. Once you feel as though you've seen enough of the shape, you can pretty confidently declare its shape. This confidence would most certainly scale with the number of different angles you got to see it from, however. If you only got to look at the shape from one angle, you couldn't really say with confidence what complex structure might be hidden on the other side.

<center><figure><img src="/assets/images/posts/QST Post/3D_Image.png" style="width:50%;height:50%;"><figcaption>Several different 2D projections of a 3D image [<a href="https://www.pdqdecide.com/post/right-solution-right-problem" target = "_blank">Image Source</a>]</figcaption></figure></center>

This is essentially the basis for MLE applied to quantum states of light. Instead of imaging some 3D shape, we are instead imaging the complex quantum state itself, and hoping to find its characteristic 'shape' in some complex phase-space. Much like in our example, our confidence in predicting the correct quantum state is directly proportional to the number of 'images' we are able to take of it from many different angles. 

# **Some Cool Visuals**
