import json

def obtener_atributos_animal(nombre, *atributos):
    # Leer el archivo JSON
    with open('animales.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Encontrar el animal específico
    animal_encontrado = next((animal for animal in data['animales'] if animal['nombre'] == nombre), None)

    if animal_encontrado:
        resultado = {atributo: animal_encontrado.get(atributo) for atributo in atributos}
        return resultado
    else:
        return f"Animal con nombre '{nombre}' no encontrado."

# Ejemplo de uso
nombre_animal_buscado = "León"
atributos = obtener_atributos_animal(nombre_animal_buscado, "tipo", "entorno")
print(atributos)