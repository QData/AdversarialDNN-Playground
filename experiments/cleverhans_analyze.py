import pandas as pd
from itertools import product

upsilon_values = [10, 15, 20, 25]
results = {}
attack_name = 'cleverhans'

for u in upsilon_values:
  try:
    df = pd.read_csv('{}-data/{}-upsilon{}.csv'.format(attack_name, attack_name, u))
    results[u] = df
  except:
    print('Couldn\'t do {} @ {}%.'.format(attack_name, u))

print('Evasion Rate:\nupsilon, cleverhans')
for u, df in results.items():
  evasion_rate = df['success'].mean() 
  print('{}, {}'.format(u, evasion_rate))

print('Times:\nupsilon, cleverhans')
for u, df in results.items():
  avg_time = df[df.success == 1]['time'].mean() 
  print('{}, {}'.format(u, avg_time))

