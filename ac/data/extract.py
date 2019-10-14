import sqlite3
import os.path
import numpy as np
import sparse

CHUNK_SIZE = int(1e6)
RELATION_COUNT = 15

def resolve_relations(db_file, rel_file, meta_file, id_file):
	"""
	"""
	conn = open_db_connection(db_file)
	c = conn.cursor()

	# load or compute unique IDs
	if os.path.isfile(meta_file):
		meta = np.load(meta_file)
		off = meta[0]
		num_unique = meta[1]
		unique_ids = np.load(id_file)
	else:
		off = 0
		c.execute("SELECT DISTINCT event1_id FROM Relations;")
		event_ids = set(c.fetchall())
		for id2 in c.execute("SELECT event2_id FROM Relations;"):
			if not id2 in event_ids:
				event_ids.add(id2)
		unique_ids = np.char.array(list(event_ids))
		num_unique = len(event_ids)
		np.save(id_file, unique_ids)
		np.save(meta_file, np.array([off, num_unique]))

	id_lookup = dict()
	for i, id_entr in enumerate(unique_ids):
		id_lookup[id_entr[0]] = i
	
	# load or compute (compressed) relations
	if os.path.isfile(rel_file):
		relations = sparse.load_npz(rel_file)
	else:
		relations = sparse.DOK((num_unique, num_unique, RELATION_COUNT), dtype=np.float32)
		for row in c.execute("SELECT * FROM Relations;"):
			id_out = row[1]
			id_in = row[2]
			relations[id_lookup[id_out], id_lookup[id_in], :] = row[3:]
		relations = sparse.COO(relations)
		sparse.save_npz(rel_file, relations)
	
	conn.close()

def extract_relation_ind(rel_file, rel_index_file):
	"""
	Save index array for nonzero entries of the 
	sparse relation tensor at rel_file.
	"""
	relations = sparse.load_npz(rel_file)
	relation_ind = sparse.argwhere(relations > 0)
	np.save(rel_index_file, relation_ind)

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
