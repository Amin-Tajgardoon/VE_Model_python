'''
Created on Oct 25, 2016

@author: mot16
'''

from os import listdir
from subprocess import call


dataDir = "C:\\Users\\mot16\\projects\\Proposal 1374\\GSK-108134\\R_analysis\\"

for f in listdir(dataDir):
    call(["pgf", "--db", "gsk", "--pw", "amin3521", "--schema", "public", "csv", dataDir + f])



