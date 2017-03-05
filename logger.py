import logging

class Logger():
    __carriage_return = "\r"
    __new_line = "\n"
    __empty_string = ""
    __colon = ":"

    def __init__(self):
        logging.basicConfig(format='[%(asctime)s] : %(levelname)s : %(message)s', level=logging.INFO)

    def new_line(self):
        print(self.__new_line)

    def println(self, msg):
        logging.info(msg)

    def print(self, msg):
        output = self.__carriage_return + msg
        print(output, end=self.__empty_string)

    def print_time_taken(self, msg, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        time_taken = "%d:%02d:%.3f" % (h, m, s)
        logging.info(msg + self.__colon + " " + time_taken)

