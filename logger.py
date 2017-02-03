import logging

class Logger:
    __carriage_return = "\r"
    __new_line = "\n"
    __empty_string = ""

    def __init__(self):
        logging.basicConfig(format='[%(asctime)s] : %(levelname)s : %(message)s', level=logging.INFO)

    def new_line(self):
        print(self.__new_line)

    def println(self, msg):
        logging.info(msg)

    def print(self, msg):
        output = self.__carriage_return + msg
        print(output, end=self.__empty_string)
