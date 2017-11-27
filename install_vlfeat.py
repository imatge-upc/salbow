import os
from IPython import embed

print "----------------------------"
print "Installing VLFEAT for python"
print "----------------------------"

# install vlfeat for python2
#cmd = 'pip install vlfeat-ctypes'
#print cmd
#os.system(cmd)
import vlfeat
cmd = "cp lib/libvl.so {}/".format( vlfeat.__path__[0] )
print cmd
os.system( cmd )

cmd = "cp lib/kmeans.py {}/kmeans.py".format( vlfeat.__path__[0] )

#print cmd
os.system(cmd)

print "DONE!"
print "----------------------------"
