import os
import glob

def abs_path(p):
	"""
	Transform an UNIX-style relative path to a 
	platform-independent absolute path.
	"""
	root = os.getcwd()
	return os.path.join(root, *tuple(p.split("/")))

def find_pickles(abs_p):
	"""
	Given absolute directory path abs_p, return a
	list of all (absolute) paths of *.pickle files in it. 
	"""
	return glob.glob(os.path.join(abs_p, "*.pickle"))