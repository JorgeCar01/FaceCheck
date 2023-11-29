import pandas as pd
import sqlite3 as sql

# TASK 1

# Connects to the database
print("Loading the Database")
con = sql.connect('students.db')
cur = con.cursor()
print("Connected")

# query = '''
#             CREATE TABLE students(
#                 name TEXT,
#                 sid INTEGER,
#                 email TEXT,
#                 PRIMARY KEY(sid)
#             );
#         '''
# t = cur.execute(query)


query = '''
            INSERT INTO students(name, sid, email)
            VALUES ('Jorge Carranza Pena', 20563986, 'jorge.carranzapena01@utrgv.edu')
        '''
t = cur.execute(query)

query = '''
            INSERT INTO students(name, sid, email)
            VALUES ('Daniel Garcia', 20535048, 'daniel.garcia39@utrgv.edu')
        '''
t = cur.execute(query)

query = '''
            INSERT INTO students(name, sid, email)
            VALUES ('Ricardo Barsenas', 20482580, 'ricardo.barsenas01@utrgv.edu')
        '''
t = cur.execute(query)

query = '''
            INSERT INTO students(name, sid, email)
            VALUES ('Daniel Alejos', 20490725, 'daniel.alejos02@utrgv.edu')
        '''
t = cur.execute(query)
con.commit()
t = cur.execute('SELECT * FROM students')
names = list(map(lambda x: x[0], t.description))
print('Student Table')
print(names)
print('--------------------')
for row in t : print(row)
print('--------------------')

con.close()