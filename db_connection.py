import configparser
import pymssql

class DbConnection:
    __config_file_name = "general.cfg"

    def get_config(self):
        return self.__config

    def read_config(self):
        #get database config
        self.__config = configparser.ConfigParser()
        self.__config.read(self.__config_file_name)

    def make_connection(self):
        self.__connection = pymssql.connect(server=self.__config.get('AzureDB', 'server_URI'), user=self.__config.get('AzureDB', 'username'), password=self.__config.get('AzureDB', 'password'), database=self.__config.get('AzureDB', 'database_1'))

    def connect(self):
        self.read_config()
        self.make_connection()
        self.__cursor = self.__connection.cursor()
        return self.__cursor
