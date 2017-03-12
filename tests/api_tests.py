import unittest

import api
from api import API

#python3 -m unittest tests.extractor_tests
class APITests(unittest.TestCase):

    def test_api_get_length(self):
        api = API()
        app = api.get_test_app()

        self.assertTrue(app.get().content_length > 0)

    def test_api_get_status(self):
        api = API()
        app = api.get_test_app()

        self.assertTrue(app.get().status_code == 200)
