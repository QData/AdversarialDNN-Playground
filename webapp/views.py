from webapp import app
from flask import render_template, request
from os import listdir

from webapp.models import l1_model

@app.route('/')
@app.route('/index')
def index():
  print(listdir())
  mnist_filename='.\webapp\models\mnist-model.meta'
  l1_model.setup(mnist_filename)
  return render_template('index.html', title='Home')
  
@app.route('/run_adversary', methods=['POST'])
def run_adversary():
  model_name    = request.form['model_name']
  upsilon_value = request.form['upsilon_value']
  
  if model_name == 'L1':
    # Perform tensor flow request
    l1_model.l1_attack('2', 7, upsilon_value)
#  else:
#    pass
    
  return "Model:{}\nUpsilon:{}".format(request.form['model_name'], request.form['upsilon_value'])