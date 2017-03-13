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

        post = app.post('/resume2entity', data={'file': (io.BytesIO(b'Test Content'), 'test.pdf')})
        self.assertTrue(post.status_code == 200)

    def test_api_post_returns_xml(self):
        api = API()
        app = api.get_test_app()

        post = app.post('/resume2entity', data={'file': (io.BytesIO(b'Test Content'), 'test.pdf')})
        self.assertTrue(post.content_type == "application/xml")

    def test_api_post_returns_data(self):
        api = API()
        app = api.get_test_app()

        post = app.post('/resume2entity', data={'file': (io.BytesIO(b'Test Content'), 'test.pdf')})
        self.assertTrue(len(post.data) > 0)

    def test_api_post_disallow_invalid_ext(self):
        api = API()
        app = api.get_test_app()
        expected_content_type = 'text/html; charset=utf-8'

        post = app.post('/resume2entity', data={'file': (io.BytesIO(b'Test Content'), 'test.exe')})

        self.assertEqual(post.status_code, 406)
        self.assertTrue(len(post.data) > 0)
        self.assertEqual(post.content_type, expected_content_type)

    def test_api_post_check_valid_filetypes(self):
        api = API()
        app = api.get_test_app()

        post = app.post('/resume2entity', data={'file': (io.BytesIO(b'Test Content'), 'test.pdf')})
        self.assertEqual(post.status_code, 200)
        post = app.post('/resume2entity', data={'file': (io.BytesIO(b'Test Content'), 'test.doc')})
        self.assertEqual(post.status_code, 200)
        post = app.post('/resume2entity', data={'file': (io.BytesIO(b'Test Content'), 'test.docx')})
        self.assertEqual(post.status_code, 200)
