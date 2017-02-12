import os
from app import app
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import foreground_extraction as fe
import cv2 

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.static_folder, filename))
            img = fe.extract_foreground(app.static_folder + '/' + filename)
            new_filename = filename.split('.')[0] + '.png'
            cv2.imwrite(app.static_folder + '/' + new_filename, img)
            os.system('rm ' + app.static_folder + '/' + filename)
            # file.save(os.path.join(app.static_folder, filename))
            return redirect(url_for('uploaded_file',
                                    filename=new_filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.static_folder,
                               filename)