import pdb
class TokenIterable:
    def __init__(self, doc, window_size):
        self.document = doc
        self.window_size = window_size

        self.last_line_idx_doc = len(doc) - 1
        # start at first line and first token
        # format: (token_idx, line_idx) and zero based indexes
        self.current_token_idx = (0, 0)
        self.current_line_size = len(doc[0])

    def __iter__(self):
        return self

    def __next__(self):
        # end of document and end of line
        if self.current_token_idx[0] == self.current_line_size and self.current_token_idx[1] == self.last_line_idx_doc:
            raise StopIteration

        else:
            # reached end of line
            if self.current_token_idx[0] == self.current_line_size:


                # first token of new line
                self.current_token_idx = (0, self.current_token_idx[1]+1)
                self.current_line_size = len(self.document[self.current_token_idx[1]])

                return_token = self.token_window()


                return return_token

            # move index along same line
            else:
                return_token = self.token_window()
                self.current_token_idx = (self.current_token_idx[0]+1, self.current_token_idx[1])
                return return_token

    def token_window(self):
        """
        if (self.current_token_idx[1] == 1 and self.current_token_idx[0] == 1):
            pdb.set_trace()
            """

        window = []
        # collected fragments are assembled into a window after gathering across lines
        token_fragments = []
        # trailing window fits behind current index on line
        if (self.current_token_idx[0]+1) >= self.window_size:
            window_start_idx = max(0, (self.current_token_idx[0]+1)- self.window_size)
            window = self.document[self.current_token_idx[1]][window_start_idx:self.current_token_idx[0]+1]
        else: # window travels between lines
            if self.current_token_idx[1] != 0: # is not first line of doc
                # keep count of how many tokens left to gather
                window_remaining = self.window_size
                current_line_idx = self.current_token_idx[1]

                switched_line = False
                while window_remaining != 0: # stop when window is filled
                    if current_line_idx > -1:
                        if not switched_line: # on same line start index at 0
                            token_fragments.append(self.document[current_line_idx][0:self.current_token_idx[0]+1])
                            window_remaining -= self.current_token_idx[0]
                        else: # not on same line 
                            current_line_size = len(self.document[current_line_idx])
                            new_line_start_idx = max(0, (current_line_size+1)- window_remaining)
                            token_fragments.insert(0, self.document[current_line_idx][max(0, new_line_start_idx):current_line_size+1])
                            window_remaining -= min(window_remaining, current_line_size)
                        current_line_idx -= 1
                        switched_line = True
                    else: # reached beginning of document
                        for x in range(0, window_remaining):
                            token_fragments.insert(0, (None, None)) # insert none to beginning of list
                            window_remaining -= 1
            else: # is first line of doc
                window_remaining = self.window_size
                have_tokens = self.document[self.current_token_idx[1]][0:self.current_token_idx[0]+1]
                for token_idx, token in enumerate(have_tokens):
                    token_fragments.append(token)
                    window_remaining -= 1

                for x in range(0, window_remaining):
                    token_fragments.insert(0, (None, None)) # insert none to beginning of list

        # concatenate fragments
        for fragment in token_fragments:
            window.extend(fragment)

        return window


