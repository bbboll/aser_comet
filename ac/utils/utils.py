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

def any_missing_file(path_list):
	"""
	Check if any of the given paths is not a file.
	"""
	out = False
	for p in path_list:
		if not os.path.isfile(p):
			out = True
	return out