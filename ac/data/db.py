import sqlite3

CHUNK_SIZE = int(1e6)

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

def get_eventuality_text(c, id_str):
	"""
	Find the text for the Eventuality with given id in the db.
	"""
	query = "SELECT words FROM Eventualities WHERE _id=?"
	c.execute(query, (id_str,))
	res = c.fetchall()
	if len(res) == 0:
		print("There is no Eventuality with id {} in this db.".format(id_str))
		exit()
	return res[0][0]