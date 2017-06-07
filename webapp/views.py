from webapp import app
from flask import render_template, request
from os import listdir

import json
import numpy as np

from webapp.models import l0_model, linf_model

with open('./webapp/models/seeds.json') as f:
  mnist_data = json.load(f)

@app.route('/')
@app.route('/index')
def index():
  return render_template('index.html', title='Home')


@app.route('/jsma')
def jsma():
  mnist_filename='./webapp/models/mnist-model.meta'
  l0_model.setup(mnist_filename)
  return render_template('jsma.html', 
      title='JSMA',
      model_name="jsma",
      targeted=True,
      display_names=sorted(mnist_data.keys()))
  
@app.route('/l_inf')
def l_inf():
  mnist_filename='./webapp/models/mnist-model.meta'
  linf_model.setup(mnist_filename)
  return render_template('fastgradientsign.html', 
      title='L_infinity Norm', 
      model_name="Linf",
      display_names=sorted(mnist_data.keys()))
  
@app.route('/fjsma')
def fjsma_handler():
  mnist_filename='./webapp/models/mnist-model.meta'
  l0_model.setup(mnist_filename)
  return render_template('fjsma.html', 
      title='FJSMA', 
      model_name="fjsma", 
      targeted=True,
      display_names=sorted(mnist_data.keys()))
  
@app.route('/run_adversary', methods=['POST'])
def run_adversary():
  print('Starting adversary generation')
  model_name    = request.form['model_name']

  seed_image    = np.array(mnist_data[request.form['sample']]['image'], ndmin=2)
  seed_class    = int(mnist_data[request.form['sample']]['class'])
  print(model_name)
  
  if model_name == 'jsma':
    # Perform tensor flow request
    upsilon_value = request.form['attack_param']
    target_value  = int(request.form['target'])
    print('Performing the jsma L0 attack from {} to {}'.format(seed_class, target_value))
    adversary_class, adv_example, adv_likelihoods = l0_model.attack(seed_image, target_value, upsilon_value)
    #return "Model:{}\nUpsilon:{}".format(request.form['model_name'], request.form['upsilon_value'])
  if model_name == 'fjsma':
    # Perform tensor flow request
    upsilon_value = request.form['attack_param']
    target_value  = int(request.form['target'])
    print('Performing the fjsma L0 attack from {} to {}'.format(seed_class, target_value))

    adversary_class, adv_example, adv_likelihoods = l0_model.attack(seed_image, target_value, upsilon_value, fast=True)
    print('hi')

  elif model_name =='Linf':
    epsilon_value = request.form['attack_param']
    adversary_class, adv_example, adv_likelihoods = linf_model.fgsm(seed_image, seed_class, epsilon_value)
    
  print('New adversary is classified as {}'.format(adversary_class))
  ret_val = {
              'adversary_class':str(adversary_class), 
              'image_data': [{
                  'z' : list(reversed(adv_example.tolist())) if adv_example is not None else '',
                  'type': 'heatmap',
                  'colorscale': [
                      ['0.0',            'rgb(0.00,0.00,0.00)'],
                      ['0.111111111111', 'rgb(28.44,28.44,28.44'],
                      ['0.222222222222', 'rgb(56.89,56.89,56.89)'],
                      ['0.333333333333', 'rgb(85.33,85.33,85.33)'],
                      ['0.444444444444', 'rgb(113.78,113.78,113.78)'],
                      ['0.555555555556', 'rgb(142.22,142.22,142.22)'],
                      ['0.666666666667', 'rgb(170.67,170.67,170.67)'],
                      ['0.777777777778', 'rgb(199.11,199.11,199.11)'],
                      ['0.888888888889', 'rgb(227.56,227.56,227.56)'],
                      ['1.0',            'rgb(256.00,256.00,256.00)']
                    ],
                  'showscale':'false',
                  'showlegend':'false',
                }],
                'likelihood_data': [{
                  'x':list(range(10)),
                  'y':[float(x) for x in adv_likelihoods],
                  'type':'bar'
                }],
            }
  return json.dumps(ret_val)
    
@app.route('/get_normal', methods=['POST'])
def normal_sample():
  sample_id = request.form['sample_id']

  with open('./webapp/models/seeds.json') as f:
    mnist_data = json.load(f)

  orig = np.array(mnist_data[sample_id]['image'], ndmin=2).reshape((28, 28))
  ret_val = {
    'image_data':[{
      'z':list(reversed(orig.tolist())),
      'type':'heatmap',
      'colorscale': [
          ['0.0',            'rgb(0.00,0.00,0.00)'],
          ['0.111111111111', 'rgb(28.44,28.44,28.44'],
          ['0.222222222222', 'rgb(56.89,56.89,56.89)'],
          ['0.333333333333', 'rgb(85.33,85.33,85.33)'],
          ['0.444444444444', 'rgb(113.78,113.78,113.78)'],
          ['0.555555555556', 'rgb(142.22,142.22,142.22)'],
          ['0.666666666667', 'rgb(170.67,170.67,170.67)'],
          ['0.777777777778', 'rgb(199.11,199.11,199.11)'],
          ['0.888888888889', 'rgb(227.56,227.56,227.56)'],
          ['1.0',            'rgb(256.00,256.00,256.00)']
        ],
    }],
    'likelihood_data':[{
      'y':[float(y) for y in mnist_data[sample_id]['likelihoods']],
      'x':list(range(10)),
      'type':'bar'
    }]
  }

  return json.dumps(ret_val)
  
