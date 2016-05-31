import os
import sys

#python exp_script.py val
#val is 0 or 1, 0 is vector version of highway network, 1 is convolutional highway network

#Highway network training:
    #Usage: python training_highway.py -data AAA -p BB -saving...
    # -data: mnist
    # -saving: log & model saving file
    # -dim: dimension of highway layers
    # -shared: 1 if shared, 0 otherwise, 2 if both


command0 = 'python training_highway.py -data mnist -dim 50 -shared 0 -saving highway_mnist_dim50_shared0'
command1 = 'python training_conv.py -data mnist -dim 16 -shared 0 -saving convhighway_mnist_dim16_shared0'

if sys.argv[1] == '0':
    command = command0
else: command = command1
print command
os.system(command)