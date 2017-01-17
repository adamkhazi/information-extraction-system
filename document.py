class TokenIterable:
    def __init__(self, doc, window_size):
        self.document = doc
        self.window_size = window_size

        self.last_line_idx = len(doc) - 1
        # start at first line and first token
        # format: (token_idx, line_idx)
        self.current_token_idx = (0, 0)
        self.current_line_size = len(doc[0])

    def __iter__(self):
        return self

    def __next__(self):
        # end of document and end of line
        if self.current_token_idx[0] == self.current_line_size and self.current_token_idx[1] == self.last_line_idx:
            raise StopIteration
        else:
            # reached end of line
            if self.current_token_idx[0] == self.current_line_size:
                print("hit end of line")

                # first token of new line
                self.current_token_idx = (0, self.current_token_idx[1]+1)
                self.current_line_size = len(self.document[self.current_token_idx[1]])

                return_token = self.document[self.current_token_idx[1]][self.current_token_idx[0]]

                return return_token
            # move index along same line
            else:
                print("incrementing token along line")
                return_token = self.document[self.current_token_idx[1]][self.current_token_idx[0]]
                self.current_token_idx = (self.current_token_idx[0]+1, self.current_token_idx[1])
                return return_token

