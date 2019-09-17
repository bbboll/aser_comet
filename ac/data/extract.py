import sqlite3
import os.path
import numpy as np
import sparse

CHUNK_SIZE = int(1e6)

def resolve_relations(db_file, rel_file, meta_file, id_file):
	"""
	"""
	conn = open_db_connection(db_file)
	c = conn.cursor()

	# load current offset
	if os.path.isfile(meta_file):
		meta = np.load(meta_file)
		off = meta[0]
		num_unique = meta[1]
	else:
		off = 0
		c.execute("SELECT DISTINCT event1_id FROM Relations;")
		event_ids = set(c.fetchall())
		print("fetched first")
		for id2 in c.execute("SELECT event2_id FROM Relations;"):
			if not id2 in event_ids:
				event_ids.add(id2)
		print("fetched second")
		unique_ids = list(event_ids)
		num_unique = len(event_ids)
		np.save(id_file, np.char.array(unique_ids))
		np.save(meta_file, np.array([off, num_unique]))

	# load (compressed) relations
	# relations = np.load()

	conn.close()

def open_db_connection(db_file):
	"""
	"""
	return sqlite3.connect(db_file)

def load_eventualities(c, offset=0, limit=CHUNK_SIZE):
	"""
	"""
	query = """
			SELECT words, frequency 
			FROM Eventualities 
			LIMIT ? OFFSET ?;
			"""
	data = []
	for row in c.execute(query, (limit, offset)):
		data.append(row)
	return data

def load_relations(c, offset=0, limit=CHUNK_SIZE):
	"""
	"""
	query = """
			SELECT *
			FROM Relations 
			LIMIT ? OFFSET ?;
			"""
	data = []
	for row in c.execute(query, (limit, offset)):
		data.append(row)
	return data
