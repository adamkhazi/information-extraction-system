import unittest
import io

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

    def test_api_post_status(self):
        api = API()
        app = api.get_test_app()

        post = app.post('/resume2entity', data={'file': (io.BytesIO(b'everyone'), 'test.pdf')})
        self.assertTrue(post.status_code == 200)

    def test_api_post_returns_xml(self):
        api = API()
        app = api.get_test_app()

        post = app.post('/resume2entity', data={'file': (io.BytesIO(b'everyone'), 'test.pdf')})
        self.assertTrue(post.content_type == "application/xml")

    def test_api_post_returns_data(self):
        api = API()
        app = api.get_test_app()

        post = app.post('/resume2entity', data={'file': (io.BytesIO(b'everyone'), 'test.pdf')})
        self.assertTrue(len(post.data) > 0)
