def load_data(filename):
    #Generador que lee un archivo línea por línea y devuelve cada producto como una tupla."""
    with open(filename, mode='r', encoding='utf-8') as file:
        header = file.readline().strip().split(",")  # Leer encabezado

        for line in file:
            value = line.strip().split(",")
            product = {"id":value[0], "name":value[1], "category":value[2],"price":value[3]}
            yield product  # Devuelve cada fila como una tupla


def count_fields(generator, header, field, value):
    #Cuenta cuántas veces aparece un valor en un campo determinado del archivo."""
    if field not in header:
        raise ValueError(f"El campo '{field}' no existe en el archivo.")

    indice = header.index(field)
    return sum(1 for row in generator if value[indice] == value)

# Leer encabezado
with open("productos.csv", mode='r', encoding='utf-8') as f:
    header = f.readline().strip().split(",")

# Contar cuántos productos pertenecen a la categoría 'Electronica'
generador_productos = load_data("productos.csv")
counting = count_fields(generador_productos, header, "categoria", "Electronica")

# Mostrar el resultado
print(f"Número de productos en la categoría 'Electronica': {counting}")

