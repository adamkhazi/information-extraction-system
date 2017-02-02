url = "http://rcparser.azurewebsites.net/ParserService.asmx/ParseAjax"
params = {"filename": "test_cv.docx",
        "fileformat": "docx",
        "UploadedFile": ("test_cv.docx", open("test_cv.docx", "rb")),
        }
