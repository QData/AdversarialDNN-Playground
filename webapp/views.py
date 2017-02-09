from webapp import app
from flask import render_template, request

@app.route('/')
@app.route('/index')
def index():
  return render_template('index.html', title='Home')
  
@app.route('/run_adversary', methods=['POST'])
def run_adversary():
  model_name    = request.form['model_name']
  upsilon_value = request.form['upsilon_value']
  
  if model_name == 'L1':
    # Perform tensor flow request
  else:
    # error!
    pass
    
  return "Model:{}\nUpsilon:{}".format(request.form['model_name'], request.form['upsilon_value'])