import sys
import pandas as pd
from itertools import product

prefixes = ['fjsma']
upsilon_values = [10, 15, 20, 25]
results = {}
k=sys.argv[1] #'5'
for prefix, u in product(prefixes, upsilon_values):
  if u not in results:
    results[u] = {}

  try:
    s = 'fjsma-k{2}-data/{0}-upsilon{1}-k{2}.csv'.format(prefix, u, k)
    print(s)
    df = pd.read_csv(s)
    results[u][prefix] = df
  except:
    print('Couldn\'t do {} @ {}%.'.format(prefix, u))


print('upsilon, fjsma')
for u, row in results.items():
  evasion_rates = [str(float(row[lbl]['success'].mean())) for lbl in prefixes]
  print(str(u) + ', ' + (','.join(evasion_rates)))

print('\nupsilon, fjsma')
for u, row in results.items():
  runtimes = [str(float(row[lbl][row[lbl].success==1]['time'].mean())) for lbl in prefixes]
  print(str(u) + ', ' + (','.join(runtimes)))
