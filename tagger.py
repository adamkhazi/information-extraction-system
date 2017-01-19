import untangle
import pdb
from nltk.tokenize import RegexpTokenizer

class Tagger:
    def get_job_titles(xml):
        return job_title_list

    def match_label(self, doc, label_str, match_tag):
        length_of_lines = [len(line) for line in doc]
        content_flat_list = [item for sublist in doc for item in sublist]

        rtokenizer = RegexpTokenizer(r'\w+')
        label_str = label_str.lower() #lowercase for comparison
        label_str = rtokenizer.tokenize(label_str)
        label_str_len = len(label_str)

        content_flat_list_len = len(content_flat_list)

        idx = label_str_len-1
        while idx < content_flat_list_len:
            print(idx)
            trailing_window_len = idx - (label_str_len-1)
            comparison = content_flat_list[trailing_window_len:idx+1]
            comparison = [item.lower() for item in comparison]
            #pdb.set_trace()
            idx += 1
            if comparison == label_str:
                #found
                print("found something================")
                for matches_idx in range(trailing_window_len, idx):
                    print(content_flat_list[matches_idx])
                    content_flat_list[matches_idx] = (doc[matches_idx], match_tag)
                idx += (label_str_len-1)

        # re create 2d list

