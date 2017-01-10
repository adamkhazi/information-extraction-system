import unittest
import configparser

import db_connection
from db_connection import DbConnection

class DbConnectionTestsi(unittest.TestCase):

    def test_config_parser(self):
        db_con = DbConnection()
        db_con.read_config()
        config = db_con.get_config()

        #config is of the right type
        self.assertIsInstance(config.get('AzureDB', 'server_URI'), str)
        self.assertIsInstance(config.get('AzureDB', 'username'), str)
        self.assertIsInstance(config.get('AzureDB', 'password'), str)
        self.assertIsInstance(config.get('AzureDB', 'database_1'), str)
        
        # config is not empty
        self.assertTrue(len(config.get('AzureDB', 'server_URI')) > 0)
        self.assertTrue(len(config.get('AzureDB', 'username')) > 0)
        self.assertTrue(len(config.get('AzureDB', 'password')) > 0)
        self.assertTrue(len(config.get('AzureDB', 'database_1')) > 0)

    def test_connection_to_db(self):
        db_con = DbConnection()

        # if connect() function call raises exception unit test fails
        conn = db_con.connect()
