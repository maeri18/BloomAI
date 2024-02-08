import sys
import os
parent = os.path.dirname(__file__)
mod = os.path.join(parent, "generate")
sys.path.append(mod)
from flask import Flask, render_template 
from generate.generate_from_saved_cpu import *
from generate.generate_gif import *
from generate.drawing import *
import threading

app = Flask(__name__)
 
@app.route('/')
def home():
    return render_template('image_render.html')

@app.route('/generate')
def generate(): 
    generate_ID()
    return render_template('image_render_gen.html', genID = "./static/Images/ID.png")

@app.route('/prev')
def prev():
    return render_template('image_render_prev.html', prevID = "./static/Images/ID.png")

@app.route('/gif')
def gif():
    #generate_gif()
    return render_template('image_render_gif.html', gifID = "./static/Images/gif.gif")

@app.route('/drawing')
def drawing_gen():
    drawing()
    return render_template('image_render_draw.html', drawingID = "./static/Images/drawingID.png")

if __name__ == '__main__':
    import socket
    socket.setdefaulttimeout(30)
    app.run(debug=True, port=9000)