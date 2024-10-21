import os
import cv2
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    operation_selection = request.form['image_type_selection']
    image_file = request.files['file']
    filename = secure_filename(image_file.filename)
    reading_file_data = image_file.read()
    image_array = np.frombuffer(reading_file_data, dtype='uint8')
    decode_array_to_img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    if operation_selection == 'gray':
        file_data = make_grayscale(decode_array_to_img)
    elif operation_selection == 'sketch':
        file_data = image_sketch(decode_array_to_img)
    elif operation_selection == 'oil': 
        file_data = oil_effect(decode_array_to_img)
    elif operation_selection == 'rgb':
        file_data = rgb_effect(decode_array_to_img)
    elif operation_selection == 'water':
        file_data = water_color_effect(decode_array_to_img)
    elif operation_selection == 'invert':
        file_data = invert(decode_array_to_img)
    elif operation_selection == 'hdr':
        file_data = HDR(decode_array_to_img)
    else:
        print('No image')

    with open(os.path.join('static/', filename), 'wb') as f:
        f.write(file_data[1].tobytes())

    return render_template('upload.html', filename=filename)

def make_grayscale(decode_array_to_img):
    converted_gray_img = cv2.cvtColor(decode_array_to_img, cv2.COLOR_RGB2GRAY)
    output_image = cv2.imencode('.PNG', converted_gray_img)
    return output_image

def image_sketch(decode_array_to_img):
    converted_gray_img = cv2.cvtColor(decode_array_to_img, cv2.COLOR_BGR2GRAY)
    sharping_gray_img = cv2.bitwise_not(converted_gray_img)
    blur_img = cv2.GaussianBlur(sharping_gray_img, (111, 111), 0)
    sharping_blur_img = cv2.bitwise_not(blur_img)
    sketch_img = cv2.divide(converted_gray_img, sharping_blur_img, scale=256.0)
    output_image = cv2.imencode('.PNG', sketch_img)
    return output_image

def oil_effect(decode_array_to_img):
    bilateral_filtered_image = cv2.bilateralFilter(decode_array_to_img, d=9, sigmaColor=75, sigmaSpace=75)
    stylized_image = cv2.stylization(bilateral_filtered_image, sigma_s=60, sigma_r=0.07)
    output_image = cv2.imencode('.PNG', stylized_image)
    return output_image

def rgb_effect(decode_array_to_img):
    rgb_effect_img = cv2.cvtColor(decode_array_to_img, cv2.COLOR_RGB2BGR)
    output_image = cv2.imencode('.PNG', rgb_effect_img)
    return output_image

def water_color_effect(decode_array_to_img):
    water_effect = cv2.stylization(decode_array_to_img, sigma_s=60, sigma_r=0.07)
    output_image = cv2.imencode('.PNG', water_effect)
    return output_image

def invert(decode_array_to_img):
    inverted_img = cv2.bitwise_not(decode_array_to_img)
    output_image = cv2.imencode('.PNG', inverted_img)
    return output_image

def HDR(decode_array_to_img):
    HDR_img = cv2.detailEnhance(decode_array_to_img, sigma_s=12, sigma_r=0.15)
    output_image = cv2.imencode('.PNG', HDR_img)
    return output_image

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=filename))

if __name__ == "__main__":
    app.run()
