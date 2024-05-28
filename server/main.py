from flask import Flask, json
from flask import request
from flask import jsonify
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app,resources={r"/*":{"origins":"*"}})
#CORS(app)

@app.route('/employees', methods=['POST'])
def create_employee():
    data = json.loads(request.data)
    code=data.get('code')
    code=code[23:]
    image_data = base64.b64decode(code)
    image_bytes = BytesIO(image_data)
    image = Image.open(image_bytes)
    image.save('output_image.png')
    print(code)
    return jsonify({"code":code})


if __name__ == '__main__':
    app.run(port=5001)

