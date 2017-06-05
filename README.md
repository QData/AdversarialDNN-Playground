Adversarial DNN Playground
==========================

Reference Paper: 



This is Andrew Norton's capstone research directory.  The goal is to perform a similar function to Google's Neural Network Playground, but with adversarial models.  It is a web service that enables the user to visualize the creation of adversarial samples to neural networks.

Framework
---------
  - Python `Flask`-based server
    - Python backend provides access to TensorFlow
    - Integration with cleverhans is also possible
  - Front-end using JQuery and Bootstrap
    - Bootstrap for static visuals
      - Used Seiyria's [Bootstrap slider][bootstrap-slider]
    - Future: Use Plottable + D3 for visualization, instead of image output from backend.
    
    
Installation
------------

The primary requirements for this package are Python 3 with Tensorflow version 1.0.1 or greater.  The `requirements.txt` file contains a listing of the required Python packages; to install all requirements, run the following:

```
pip3 -r install requirements.txt
```

With Mac verison of pip, this should be:

```
pip3 install -r requirements.txt
```


Use:
----

Once you've downloaded the repo, run `python3 run.py` and navigate to `localhost:9000`.
  

Citation:
---------


```
@article{norton2017advplayground,
  title={Adversarial Playground: A Visualization Suite for Adversarial Sample Generation},
  author={Norton, Andrew and Qi, Yanjun},
  year={2017},
}
```

[bootstrap-slider]: https://github.com/seiyria/bootstrap-slider
