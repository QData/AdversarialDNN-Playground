#!/bin/bash

# Command line arguments of tests.py
# First: output filename
# Second: value of upsilon
# Third: Method used
# Fourth: Percentage of feature set to use for k.  (e.g. 5 would be 5% of the feature set)
#         I chose this percentage instead of a fixed number because it makes it 
#         easier (conceptually) to scale to larger feature sets.
# Set env variable $EMAIL_ADDRESS to be notified when jobs complete

### CLEVERHANS EXPERIMENTS ###
cleverhans () {
  for x in 10 15 20 25; do
    echo "Doing iteration ${x} for cleverhans"
    python tests.py cleverhans-upsilon${x}.csv ${x} cleverhans 
  done

  # Possible email update
  if [[ ! -z $EMAIL_ADDRESS ]]; then
    echo "Cleverhans data has been generated" | mail -s "Job complete! (Cleverhans)" $EMAIL_ADDRESS
  fi
}

### FJSMA EXPERIMENTS ###
fjsma () {
  k=$1 # Use top k% of the samples
  for x in 10 15 20 25; do
    echo "Doing iteration ${x} for fjsma"
    python tests.py fjsma-upsilon${x}-k${k}.csv ${x} fjsma ${k}
  done

  # Cleanup!
  mkdir fjsma-k${k}-data
  mv *.csv fjsma-k${k}-data

  # Possible email update
  if [[ ! -z $EMAIL_ADDRESS ]]; then
    python fjsma_analyze.py ${k} | mail -s "Job complete. (FJSMA k=${k})" $EMAIL_ADDRESS
  fi
}

## EDIT DOWN HERE FOR WHICH TO RUN
# fjsma 5    # Run with k = 5% of feature set
cleverhans # Run cleverhans tests
