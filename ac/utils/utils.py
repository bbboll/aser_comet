import os

def abs_path(p):
	"""
	Transform an UNIX-style relative path to a 
	platform-independent absolute path.
	"""
	root = os.getcwd()
	return os.path.join(root, *tuple(p.split("/")))