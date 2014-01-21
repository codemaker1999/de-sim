'''
Creates a live animation of [a] particle[s] moving based on
differential equations of motion. Uses scipy's odeint to
compute particle paths, and matplotlib's animation API for
visualization.
'''

################################################################################
#
#   DE computation
#

from math import pi
from random import random as r
import numpy as np
from numpy import array
from scipy.integrate import odeint

##
# Init Constants
# (do not modify, see "Setup" section below)

dt = 0.
steps = 0.

##
# Lorenz Equations

def lorenz(xyz,t=None):
  # unpack
  x,y,z = xyz
  # constants
  delta = 10.
  r     = 28.
  b     = 8/3.
  # compute step
  newx = delta*(y - x)
  newy = r*x - y - x*z
  newz = x*y - b*z
  return array([newx,newy,newz])

##
# Rossler system

def rossler(xyz,t=None):
  # unpack
  x,y,z = xyz
  # constants
  a = 0.2
  b = 0.2
  c = 5.7
  # compute step
  newx = -y-z
  newy = x + a*y
  newz = b + z*(x-c)
  return array([newx,newy,newz])

##
# Runge-Kutta Method

def rk4(f,xyz):
  newxyz = []
  for i in range(steps):
    # Keep a copy
    xyz0 = xyz.copy()
    # RK method
    a_array = dt*f(xyz)
    xyz     = xyz + a_array*(dt/2.)
    b_array = dt*f(xyz)
    xyz     = xyz + b_array*(dt/2.)
    c_array = dt*f(xyz)
    xyz     = xyz + c_array*dt
    d_array = dt*f(xyz)
    # Compute estimated component
    xyz  = xyz0 + (dt/6.)*(a_array + 2*b_array + 2*c_array + d_array)
    newxyz.append( xyz )
  return array(newxyz)

##
# Scipy DE intagration

def scipy_step(f,xyz):
  'one integration step'
  newxyz = odeint(f,xyz,[dt*i for i in xrange(steps)])
  return newxyz

##################################################################################
##
##  Setup - Watch out for float division
##

##########
# TIMING #
##########

# Time step
dt = 1/300.

# Number of steps per frame
steps = 30

# How many miliseconds per frame
fps = 27.
msperframe = 1000/fps

############
# PLOTTING #
############

# Function to plot
DEfunc = lorenz #lorenz rossler

# DE solving function
DEsolver = scipy_step #rk4 scipy_step

# Number of plots to draw
num_of_plots = 3

# min and max bounds on initial starting points
icmin = -10
icmax = 10

# Bounds of graph
xbnds = (-30, 30)
ybnds = (-30, 30)
zbnds = (  0, 50)

# Max number of points to display per trajectory
# (for efficiency over long times without blitting (blitting is buggy))
# Use 0 for no maximum
maxpts = 2000 #arbitrary

# factor to deviate ICs from each other for each trajectory
perterb_constant = 1/100.

#################
# OTHER OPTIONS #
#################

# Determines whether or not to automatically rotate camera
rotcam = False

# Blitting only draws to canvas when it is changed
useblitting = False

##################################################################################
###
###   Plotting
###

from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import random as r

##
# Set up figure and axis
fig = plt.figure()
ax  = p3.Axes3D(fig)
ax.set_xlim( xbnds )
ax.set_ylim( ybnds )
ax.set_zlim( zbnds )

##
# generate random initial conditions

def rand(n1,n2):
  'Generate random float between n1 and n2'
  base = r.randint(n1,n2) # int between n1 and n2
  p    = r.random()       # float between 0 and 1
  return base*p

randomIC = array([rand(icmin,icmax) for i in xrange(3)])
print 'Initial Conditions :'
data = []
for i in xrange(num_of_plots):
  # perturb ICs
  perturb = randomIC + array([r.random()*perterb_constant for i in xrange(3)])
  # print ICs
  print perturb
  # set ICs
  vals  = array([perturb])
  line, = ax.plot( [vals[0,0]] , [vals[0,1]] , [vals[0,2]] )
  # keep for computation
  data.append( [line,vals] )
print ''

##
# animation function
def animate(i):
  'iterate animation'
  ##
  # DE computation
  global data
  newdata = []
  lines = []
  for l,v in data:
    # get next step
    old_xyz = v[-1,:]
    new_xyz = DEsolver(DEfunc,old_xyz)
    # add new step to vals
    v = np.append(v,new_xyz,axis=0)
    # trim data
    if maxpts != 0 and len(v) > maxpts:
      v = v[-maxpts:]
    # update plot data
    l.set_data(v[:,0],v[:,1]) # no 3d version of set_data...
    l.set_3d_properties(v[:,2]) # so use this method for z
    newdata.append( [l,v] )
    lines.append(l)
  data = newdata
  ##
  # Camera Rotation
  if rotcam: ax.view_init(elev=25., azim=0.4*i)
  #
  return lines

##
# create animation
anim = animation.FuncAnimation(fig, animate, interval=msperframe, blit=useblitting)

##
# Show animation
try:
  # matplotlib breaks when you exit it
  plt.show()
except AttributeError:
  pass
