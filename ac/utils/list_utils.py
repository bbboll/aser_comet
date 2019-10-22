import itertools

def flatten_list(l):
	"""
	Flatten a list of lists.
	"""
	return list(itertools.chain.from_iterable(l))

def chunk_range(size_total, chunksize):
	"""
	Generate a list of (start,end) index pairs for chunks of size_total.
	"""
	i = 0
	chunks = []
	while i < size_total:
		chunks.append((i,min(i+chunksize, size_total)))
		i += chunksize
	return chunks