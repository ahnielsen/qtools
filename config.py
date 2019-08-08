# Configuration parameters
version = '0.1'
verbose = True

# Define function that will print output if verbose 
vprint = print if verbose else lambda *a, **k: None
