Adversarial DNN Playground
==========================

Reference Paper:

"Adversarial Playground: A Visualization Suite for Adversarial Sample Generation", Norton, Andrew and Qi, Yanjun, http://arxiv.org/abs/1706.01763

This is Andrew Norton's capstone research work.  The goal is to perform a similar function to Google's TensorFlow Playground, but for evasion attacks in adversiaral machine learning.  It is a web service that enables the user to visualize the creation of adversarial samples to neural networks.

Screenshots and Demo
--------------------

Information regarding the various settings for each attack model may be found the [project slide set](https://github.com/QData/AdversarialDNN-Playground/blob/master/presentation.pdf) (see especially the *System Demonstration* section).


Installation
------------

There are git submodules in this repository; to clone all the needed files, please use:

```
git clone --recursive https://github.com/QData/AdversarialDNN-Playground.git
```

The primary requirements for this package are Python 3 with Tensorflow version 1.0.1 or greater.  The `requirements.txt` file contains a listing of the required Python packages; to install all requirements, run the following:

```
pip3 -r install requirements.txt
```

If the above command does not work, use the following:

```
pip3 install -r requirements.txt
```

Or use the following instead if need to sudo:
```
sudo -H pip  install -r requirements.txt
```

Use:
----

Once you've downloaded the repo, run `python3 run.py` and navigate to `localhost:9000`.

### Modifying Seed Images
By default, we give the user the option of 11 seed images (one from each class 0 through 9, and one misclassified instance from the "9" class).  However, you may desire to select different images for your own instance of this tool.  It is quite easy to do so via the `json_gen.py` script in the [`utils`](https://github.com/QData/AdversarialDNN-Playground/tree/master/utils) directory.  Edit the `images_to_generate.csv` file to specify the indices into the MNIST dataset which interest you in the first column, and provide a human readable description in the second column.

After editing the `images_to_generate.csv` file, run:
```
$ python images_to_generate.csv
```

This will take a short amount of time, as it processes and classifies (using the pre-trained model) each seed image, and saves a `png` file of the image.  There are two items created as output, and they must be moved into proper locations in the `webapp` directories:
  - `seeds.json` : This contains each image and the classifier output as a JSON object; it goes in `/webapp/models`.
  - `imgs/` : This directory should be copied into the `webapp/static/` directory.

Run (or restart) the webserver, and the new options should be visible to the user.

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
