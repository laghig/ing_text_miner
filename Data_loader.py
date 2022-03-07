import MySQLdb
import mysql.connector
from mysql.connector import Error
import yaml
import sys
import os


#set the working directory
path = r"C:\Users\Giorgio\Desktop\ETH\Code"
os.chdir(path)

# Load the configuration file
if os.path.exists("config.yml"):
    with open(os.getcwd() +'\config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
else:
    print('Config file is missing.')
print(config)

print(config['mysql_db']['Host'])

# connect to the mysql database
try:
    connection = mysql.connector.connect(host=config['mysql_db']['Host'],    # your host, usually localhost
                        user=config['mysql_db']['User'],         # your username
                        passwd=config['mysql_db']['Password'],  # your password
                        db=config['mysql_db']['Database'])        # name of the data base

    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
except Error as e:
    print("Error while connecting to MySQL", e)