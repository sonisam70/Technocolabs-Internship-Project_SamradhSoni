import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, Response, request
from ASL_Camera import image_predict
import numpy as np
from PIL import Image

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model.h5'

@app.route("/", methods = ['GET','POST'])
def index():
    pred = None
    message = None
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.convert('L')
                img = img.resize((100,100))
                img = np.asarray(img)
                print(img.shape)
                img = img.reshape((1,100,100,1))
                img = img/255.0
                pred = image_predict(img)
        except:
            message = "Please upload an Image"
            return render_template('index.html', message = message)
    return render_template("index.html", pred = pred, message = message)

if __name__=='__main__':
    app.run(debug=True)