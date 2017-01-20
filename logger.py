class Logger:
    __carriage_return = "\r"
    __empty_string = ""

    def println(self, msg):
        print(msg)

    def print(self, msg):
        output = self.__carriage_return + msg
        print(output, end=self.__empty_string)
