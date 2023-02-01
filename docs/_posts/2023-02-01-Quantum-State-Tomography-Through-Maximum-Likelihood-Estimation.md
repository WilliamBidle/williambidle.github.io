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

# **What is Quantum State Tomography?**
The basic idea behind Quantum State Tomography (QST) is that for a set of measured statistical values of an identically prepared quantum state, we are able deduce a best possible guess for what that quantum state is. The idea is not too dissimilar from a CT scan of the brain, where several different 2D snapshots are layered together in order to create an accurate 3D rendering of what is actually going on.   

<center><figure><img src="/assets/images/posts/QST Post/Brain.png" style="width:50%;height:50%;"><figcaption>CT scan of a brain [<a href="https://en.wikipedia.org/wiki/Computed_tomography_of_the_head" target = "_blank">Image Source</a>]</figcaption></figure></center>


Now typically it is very easy to deduce what a quantum state will look like if you are given the state itself. However, the opposite situation is extremely difficult: If I give you what the quantum state looks like, can you tell me what the state is without any prior information? Going back to our simple metaphor of ...

# **Maximum Likelihood Estimation**
The basic approach to Maximum Likelihood Estimation (MLE) is that we are hoping to find a density matrix that has the highest probability of representing the statistics of some measured quantum state. In other words, we are looking to maximize the likelihood that a given density matrix describes the measured quadrature data given by a likelihood equation. 

Now this generally boils down to some complicated differential mathematics involving matricies (see my thesis above for details), but most of the details are honestly irrelevant for demonstrating the idea. 

<center><figure><img src="/assets/images/posts/QST Post/3D_Image.png" style="width:50%;height:50%;"><figcaption>Several different 2D projections of a 3D image [<a href="https://www.pdqdecide.com/post/right-solution-right-problem" target = "_blank">Image Source</a>]</figcaption></figure></center>

# **Some Cool Visuals**
