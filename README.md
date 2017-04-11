Adversarial DNN Playground
==========================

This is Andrew Norton's capstone research directory.  The goal is to perform a similar function to Google's Neural Network Playground, but with adversarial models.  It is a web service that enables the user to visualize the creation of adversarial samples to neural networks.

Framework
---------
  - Python `Flask`-based server
    - Python backend provides access to TensorFlow
    - Integration with cleverhans is also possible
  - Front-end using JQuery and Bootstrap
    - Bootstrap for static visuals
      - Used Seiyria's [Bootstrap slider][1]
    - Future: Use Plottable + D3 for visualization, instead of image output from backend.
    
    
Python Packages
---------------

Relevant Python 3 packages:
  - numpy/scipy stack
  - tensorflow
  - Flask ([A good Flask resource for the unfamiliar][flask-intro]--including me!)

Use:
----

Once you've downloaded the repo, run `python3 run.py` and navigate to `localhost:5000`.
  
[flask-intro]: https://github.com/seiyria/bootstrap-slider/
