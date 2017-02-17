from webapp import app
from flask import render_template, request
from os import listdir

from webapp.models import l1_model, linf_model

@app.route('/')
@app.route('/index')
def index():
  mnist_filename='.\webapp\models\mnist-model.meta'
  l1_model.setup(mnist_filename)
  return render_template('index.html', title='Home')
  
@app.route('/l_inf')
def l_inf():
  mnist_filename='.\webapp\models\mnist-model.meta'
  linf_model.setup(mnist_filename)
  return render_template('fastgradientsign.html', title='L_infinity Norm')
  
@app.route('/run_adversary', methods=['POST'])
def run_adversary():
  print('Starting adversary generation')
  model_name    = request.form['model_name']
  
  if model_name == 'L1':
    # Perform tensor flow request
    upsilon_value = request.form['upsilon_value']
    l1_model.l1_attack('2', 7, upsilon_value)
  elif model_name =='Linf':
    epsilon_value = request.form['epsilon_value']
    linf_model.fgsm('2', epsilon_value)
    
  return "Model:{}\nUpsilon:{}".format(request.form['model_name'], request.form['upsilon_value'])