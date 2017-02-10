Adversarial Playground
======================

This is Andrew Norton's capstone research directory.  The goal is to perform a similar function to Google's Neural Network Playground, but with adversarial models.  It is a web service that enables the user to visualize the creation of adversarial samples to neural networks.

Goal is to submit to ICLR by the 14th.

Framework
---------
  - Python "Flask"-based server
    - Enables access to TensorFlow
    - Possibly integrate with cleverhans
  - JQuery front-end 
    - Plottable + D3 for visualization
    - Bootstrap for static visuals
      - Used Seiyria's [Bootstrap slider][1]
    
    
Python Packages
---------------

I'm not using virtualenv (for various bad reasons), so I'm making a list of the packages I've used here:
  - numpy/scipy stack
  - tensorflow (of course)
  - Flask ([A good Flask resource for the unfamiliar][2]--including me!)
  
[1]: https://github.com/seiyria/bootstrap-slider/