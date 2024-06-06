from flask import Flask, render_template, request, session, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf


app = Flask(__name__, static_folder="static")
#app.secret_key = 'your_secret_key'  
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
            return render_template('predecir.html', error="No se proporcionó ninguna imagen.")

        image_file = request.files['image']
        if image_file.filename == '':
            return render_template('predecir.html', error="El archivo de imagen no tiene un nombre válido.")

        # Establecer un nombre fijo para la imagen
        image_filename = 'temp_image.jpg'
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

        # Guardar el archivo de imagen
        image_file.save(image_path)

        model_name = request.form['model']  # Obtener el modelo seleccionado por el usuario
        predicted_results = predict(image_path, model_name)

        return render_template('predecir.html', image_path=image_filename, predicted_results=predicted_results, model_names=model_names)

    if 'image_path' in session:
        image_path = session['image_path']
    else:
        image_path = None

    return render_template('index.html', image_path=image_path, model_names=model_names)


@app.route('/m', methods=['POST'])
def get_description():
    user_input = request.form['user_input']
    # Aquí deberías integrar la llamada a Google Generative AI
    response = {"conversation": f"Descripción generada para: {user_input}"}
    return jsonify(response)

@app.route('/predecir')
@app.route('/predecir')
def predecir():
    return render_template('predecir.html', model_names=model_paths.keys())

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

    # Obtener los índices de los  resultados más probables
    cantidad_resultados = 1
    top_indices = np.argsort(result)[::-1][:cantidad_resultados]

    # Obtener las clases correspondientes a los índices
    predicted_classes = [f"{class_names[model_name][i]} - {result[i]}" for i in top_indices]

    return predicted_classes

if __name__ == '__main__':
    app.run(debug=True)
