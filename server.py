import os
from flask import Flask, render_template, request, url_for
import yolo

UPLOAD_FOLDER = os.path.join('static')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/', methods = ['GET','POST'])
@app.route('/main', methods = ['GET','POST'])
def main_page():
    result = ""
    if request.method == 'POST':
        if 'img' not in request.files:
            result = "No file part"
        file = request.files['img']
        if file.filename == '':
            result = "No file selected"
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "test.jpg")
            file.save(filepath)
            yolo.detect()
            result = os.path.join(app.config['UPLOAD_FOLDER'], "result.jpg")
            return render_template("result.html", result = result)
    return render_template("main.html", result = result)

if (__name__=='__main__'):
    app.run(debug=True, port=5000)