class Logger:
    __carriage_return = "\r"
    __new_line = "\n"
    __empty_string = ""

    def new_line(self):
        print(self.__new_line)

    def println(self, msg):
        print(msg)

    def print(self, msg):
        output = self.__carriage_return + msg
        print(output, end=self.__empty_string)
