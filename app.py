from flask import Flask, render_template, request, jsonify, url_for, session
import os
import cv2
import numpy as np
from datetime import datetime
import openai
import tensorflow as tf

# Clave de la API de OpenAI
openai_api_key = os.environ.get('sk-proj-tIHADECUoicEKPxQt032T3BlbkFJSsgKbNQalPWDlnnt674I')

app = Flask(__name__, static_folder="static")
app.secret_key = 'supersecretkey'

app.config['SESSION_COOKIE_SECURE'] = True
app.config['UPLOAD_FOLDER'] = 'static'

# Rutas a los modelos y archivos de etiquetas
model_paths = {
    'Modelo 2': {
        'model': 'segundo_modelo.tflite',
        'labels': 'segundo_modelo.txt'
    }
}

# Cargar e inicializar intérpretes para ambos modelos
interpreters = {}
input_details = {}
output_details = {}
class_names = {}

for model_name, paths in model_paths.items():
    model_path = paths['model']
    interpreters[model_name] = tf.lite.Interpreter(model_path=model_path)
    interpreters[model_name].allocate_tensors()
    input_details[model_name] = interpreters[model_name].get_input_details()
    output_details[model_name] = interpreters[model_name].get_output_details()
    with open(paths['labels'], 'r', encoding='utf-8') as f:
        class_names[model_name] = f.read().splitlines()

@app.route('/', methods=['GET', 'POST'])
def index():
    model_names = list(model_paths.keys())  # Obtener los nombres de los modelos disponibles

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predecir.html', error="No se proporcionó ninguna imagen.", model_names=model_names)

        image_file = request.files['image']
        if image_file.filename == '':
            return render_template('predecir.html', error="El archivo de imagen no tiene un nombre válido.", model_names=model_names)

        # Establecer un nombre fijo para la imagen
        image_filename = 'temp_image.jpg'
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

        # Guardar el archivo de imagen
        image_file.save(image_path)

        model_name = request.form['model']  # Obtener el modelo seleccionado por el usuario
        predicted_results = predict(image_path, model_name)

        return jsonify({
            'image_path': url_for('static', filename=image_filename) + '?t=' + datetime.now().strftime('%Y%m%d%H%M%S'),
            'predicted_results': predicted_results
        })

    return render_template('index.html', model_names=model_names)

@app.route('/m', methods=['POST'])
def chat():
    # Reiniciar la conversación si se presiona el botón de enviar una pregunta
    if request.form.get("question"):
        session["conversation"] = ""

    user_input = request.form.get("user_input")

    if user_input:
        try:
            # Procesar la pregunta recibida
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": user_input}
                ],
                api_key=openai_api_key
            )

            if response.choices:
                chatbot_response = response.choices[0].message.content

                # Agregar solo la respuesta del chatbot a la conversación
                session["conversation"] += f"DESCRIPCION DEL ANIMAL: {chatbot_response}\n"
            else:
                session["conversation"] += f"Error al hacer la petición"
        except Exception as e:
            session["conversation"] += f"Error al hacer la petición: {str(e)}\n"

@app.route('/predecir')
def predecir():
    return render_template('predecir.html', model_names=model_paths.keys())

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Invalid image file name'}), 400

    image_filename = 'temp_image.jpg'
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    image_file.save(image_path)

    model_name = request.form['model']
    predicted_results = predict(image_path, model_name)
    return jsonify({
        'image_path': url_for('static', filename=image_filename) + '?t=' + datetime.now().strftime('%Y%m%d%H%M%S'),
        'predicted_results': predicted_results
    })

def predict(image_path, model_name):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Redimensionar la imagen a 224x224 y conviértela a UINT8
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized.astype(np.uint8)
    img_resized = np.expand_dims(img_resized, axis=0)

    # Realizar la predicción
    interpreters[model_name].set_tensor(input_details[model_name][0]['index'], img_resized)
    interpreters[model_name].invoke()
    result = interpreters[model_name].get_tensor(output_details[model_name][0]['index'])[0]

    # Obtener los índices de los resultados más probables
    cantidad_resultados = 1
    top_indices = np.argsort(result)[::-1][:cantidad_resultados]

    # Obtener las clases correspondientes a los índices
    predicted_classes = [f"{class_names[model_name][i]} - {result[i]:.2f}" for i in top_indices]

    return predicted_classes

if __name__ == '__main__':
    app.run(debug=True)