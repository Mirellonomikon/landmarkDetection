import base64

import cv2
from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
import os

from landmarkDetection import landmark

app = Flask(__name__)
app.config['SECRET_KEY'] = 'landmark_detection'
app.config['UPLOADED_PHOTOS_DEST'] = 'static/uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, "Only images allowed."),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')


@app.route('/static/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


@app.route('/', methods=['GET', 'POST'])
def upload_image():  # put application's code here
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for("get_file", filename=filename)
        # Pass the image to the landmarks method
        img = landmark(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        # Save the modified image in the same folder as the uploaded images
        cv2.imwrite(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], 'modified_' + filename), img)
        # remove duplicate of original picture
        os.remove(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        # Encode the image to PNG format
        ret, png = cv2.imencode('.png', img)
        # Convert the image data to a base64-encoded string
        img_data = base64.b64encode(png.tobytes()).decode()
    else:
        file_url = None
        img_data = None
    return render_template('index.html', form=form, file_url=file_url, img_data=img_data)


if __name__ == '__main__':
    app.run()
