Adversarial DNN Playground
==========================

Reference Paper: 

"Adversarial Playground: A Visualization Suite for Adversarial Sample Generation", Norton, Andrew and Qi, Yanjun, http://arxiv.org/abs/1706.01763

This is Andrew Norton's capstone research work.  The goal is to perform a similar function to Google's TensorFlow Playground, but for evasion attacks in adversiaral machine learning.  It is a web service that enables the user to visualize the creation of adversarial samples to neural networks.

Screenshots and Demo
--------------------

We are hosting a live demo of the project at http://qdev2.cs.virginia.edu:9000.  Information regarding the various settings for each attack model may be found the [project slide set](https://github.com/QData/AdversarialDNN-Playground/blob/master/presentation.pdf) (see especially the *System Demonstration* section).
    
Installation
------------

The primary requirements for this package are Python 3 with Tensorflow version 1.0.1 or greater.  The `requirements.txt` file contains a listing of the required Python packages; to install all requirements, run the following:

```
pip3 -r install requirements.txt
```

If you are using Mac and the above does not work, use the following:

```
pip3 install -r requirements.txt
```

There are git submodules in this repository; to clone all the needed files, please use:

```
git clone --recursive git@github.com:QData/AdversarialDNN-Playground.git
```

Use:
----

Once you've downloaded the repo, run `python3 run.py` and navigate to `localhost:9000`.

Framework
---------
  - Python `Flask`-based server
    - Python backend provides access to TensorFlow
    - Integration with cleverhans is also possible
  - Front-end using JQuery and Bootstrap
    - Bootstrap for static visuals
      - Used Seiyria's [Bootstrap slider][bootstrap-slider]
    - Ploty.JS utilized for visualization

Citation:
---------

```
@article{norton2017advplayground,
  title={Adversarial Playground: A Visualization Suite for Adversarial Sample Generation},
  author={Norton, Andrew and Qi, Yanjun},
  url = {http://arxiv.org/abs/1706.01763}
  year={2017},
}
```

[bootstrap-slider]: https://github.com/seiyria/bootstrap-slider
