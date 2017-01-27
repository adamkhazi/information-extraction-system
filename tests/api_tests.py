import unittest

import api
from api import API

#python3 -m unittest tests.extractor_tests
class APITests(unittest.TestCase):
    def test_api(self):
        api = API()
        api.run()



