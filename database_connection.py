import configparser
import pymssql

class DatabaseConnection:
    __config_file_name = "general.cfg"

    def read_config(self):
        #get database config
        self.__config = configparser.ConfigParser()
        self.__config.read(self.__config_file_name)

    def make_connection(self):
        self.__connection = pymssql.connect(server=self.__config.get('AzureDB', 'server_URI'), user=self.__config.get('AzureDB', 'username'), password=self.__config.get('AzureDB', 'password'), database=self.__config.get('AzureDB', 'database_1'))

    def connect(self):
        self.read_config()
        self.make_connection()
        return self.__connection

