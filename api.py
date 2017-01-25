from flask import Flask
from flask import Response
from flask import request, redirect, request, url_for, send_from_directory, make_response
from flask import Flask, render_template

import os

from crf_suite import CrfSuite
from annotator import Annotator

class API():
    __path_output_xml = "output"
    __path_to_welcome = "index"
    __seperator = "/"

    def __init__(self):
        self.__app = Flask(__name__, template_folder='web')
        self.__app.config['ALLOWED_EXTENSIONS'] = set(['pdf', 'doc', 'docx'])
        self.__app.config['UPLOAD_FOLDER'] = 'web/uploads'

        self.__crfsuite = CrfSuite()
        self.__crfsuite.load_tagger()

    def run(self):
        self.__app.add_url_rule("/", "index", self.handle_welcome)
        self.__app.add_url_rule("/uploads/<filename>", "uploaded_file", self.uploaded_file)
        self.__app.add_url_rule("/resume2entity", "IE", self.handle_resume_post, methods=['GET', 'POST',])
        self.__app.run(host='0.0.0.0', debug=True)

    def handle_welcome(self):
        return render_template('%s.html' % self.__path_to_welcome)

    def handle_resume_post(self):
        # Get the name of the uploaded file
        file = request.files['file']
        print("hit handle method resume post")

        if file and self.__allowed_file(file.filename):
            # Save file to upload folder
            file.save(os.path.join(self.__app.config['UPLOAD_FOLDER'], file.filename))

            # use crf here
            annotator = Annotator()
            annotated_resume = annotator.annotate_using_trained_model(self.__app.config['UPLOAD_FOLDER'] + self.__seperator + file.filename)

            tagged_resume = self.__crfsuite.tag_doc(annotated_resume)

            template = render_template('%s.xml' % self.__path_output_xml, entities=tagged_resume)
            response = make_response(template)
            response.headers['Content-Type'] = 'application/xml'

            return response
            #return redirect(url_for('uploaded_file', filename=file.filename))

    # For a given file, return whether it's an allowed type or not
    def __allowed_file(self, filename):
        return '.' in filename and \
           filename.rsplit('.', 1)[1] in self.__app.config['ALLOWED_EXTENSIONS']

    # show file in browser
    def uploaded_file(self, filename):
        return send_from_directory(self.__app.config['UPLOAD_FOLDER'], filename)

    #if __name__ == '__main__':
