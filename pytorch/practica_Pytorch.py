# importamos las bibliotecas necesarias de pytorch para construir y entrenar nuestro modelo de aprendizaje automatico

# torch -> biblioteca principal de pytorch que nos permite manejar tensores (estructuras similares a matrices) y realizar operaciones matematicas sobre ellos.
import torch  

# nn -> modulo de pytorch que contiene herramientas para construir redes neuronales, como capas de neuronas.
import torch.nn as nn  

# optim -> modulo de pytorch que contiene algoritmos de optimizacion, los cuales ajustan los parametros del modelo para mejorar su precision.
import torch.optim as optim  

# torch.tensor(data) -> convierte una lista o una matriz en un tensor de pytorch, que es una estructura de datos optimizada para calculos numericos.

# creamos un tensor con los valores de entrada (x), es decir, los datos que el modelo usara como referencia.
# cada numero dentro de [[ ]] es un elemento de la matriz (matriz de 4 filas y 1 columna).
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # tensor con valores de entrada

# creamos un tensor con las salidas esperadas (y), es decir, los resultados correctos para cada entrada.
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # tensor con valores esperados de salida

# class nombredeclase(nn.module) -> definimos una clase que representa el modelo de aprendizaje.
# nn.module -> es la clase base de todos los modelos en pytorch. permite crear modelos con diferentes capas y funciones.

class modelo(nn.Module):  # creamos una clase llamada "modelo" que hereda de nn.module
    def __init__(self):  # metodo constructor que inicializa el modelo
        """
        constructor de la clase.
        en este metodo se definen los componentes del modelo de aprendizaje.
        """
        super(modelo, self).__init__()  # llamamos al constructor de la clase padre (nn.module) para inicializar correctamente el modelo.
        
        # definimos una capa de red neuronal que toma 1 entrada y produce 1 salida.
        # nn.linear(in_features, out_features) -> crea una capa completamente conectada.
        # in_features = 1 -> numero de entradas (1 porque cada dato de x tiene un solo valor).
        # out_features = 1 -> numero de salidas (1 porque queremos una unica prediccion para cada entrada).
        self.lineal = nn.Linear(1, 1)  

    # forward(x) -> metodo que define como el modelo transforma una entrada en una salida.
    def forward(self, x):  
        # funcion de propagacion hacia adelante.
        # recibe un tensor x (entrada) y lo pasa por la capa lineal para obtener la prediccion.
        return self.lineal(x)  # se aplica la transformacion lineal a la entrada y se devuelve la prediccion.

# creamos una instancia de nuestro modelo, es decir, un objeto basado en la clase modelo.
modelo = modelo()  

# la funcion de perdida mide la diferencia entre las predicciones del modelo y los valores reales.

# nn.mseloss() -> mse (mean squared error, error cuadratico medio) mide la diferencia entre las salidas predichas y las reales.
# se usa en problemas de regresion porque penaliza mas los errores grandes.
criterio = nn.MSELoss()  

# el optimizador es un algoritmo que ajusta los pesos del modelo para mejorar sus predicciones.

# optim.sgd(params, lr) -> descenso de gradiente estocastico, un metodo para minimizar la funcion de perdida.
# params = modelo.parameters() -> parametros del modelo (pesos y sesgos) que seran ajustados.
# lr = 0.01 -> tasa de aprendizaje, controla que tan rapido se ajustan los pesos del modelo.
optimizador = optim.SGD(modelo.parameters(), lr=0.01)  

# for epoca in range(1000) -> bucle que entrena el modelo durante 1000 iteraciones.
for epoca in range(1000):  
    # paso 1: calculamos las predicciones del modelo con los valores de entrada x.
    predicciones = modelo(x)  

    # paso 2: calculamos la perdida comparando las predicciones con los valores reales y.
    # criterio(predicciones, y) -> calcula la diferencia entre la salida predicha y la salida esperada.
    perdida = criterio(predicciones, y)  

    # paso 3: reiniciamos los gradientes del optimizador.
    # optimizador.zero_grad() -> borra los gradientes acumulados para evitar que se sumen en cada iteracion.
    optimizador.zero_grad()  

    # paso 4: calculamos los gradientes de la perdida con respecto a los pesos del modelo.
    # perdida.backward() -> calcula la derivada de la perdida para saber en que direccion ajustar los pesos.
    perdida.backward()  

    # paso 5: ajustamos los pesos del modelo utilizando los gradientes calculados.
    # optimizador.step() -> ajusta los parametros del modelo usando la informacion de los gradientes.
    optimizador.step()  

    # cada 100 iteraciones imprimimos la perdida para ver como mejora el modelo.
    if (epoca+1) % 100 == 0:  
        print(f'epoca [{epoca+1}/1000], perdida: {perdida.item():.4f}')  # muestra la perdida con 4 decimales.

# probamos el modelo con un nuevo valor de entrada (5.0) para ver su prediccion.
entrada_nueva = torch.tensor([[5.0]])  # creamos un tensor con el valor 5.0
salida_predicha = modelo(entrada_nueva)  # obtenemos la prediccion del modelo para esta entrada

# mostramos el resultado predicho.
# .item() -> extrae el valor del tensor y lo convierte en un numero normal de python.
print(f'para una entrada de 5.0, la salida predicha es: {salida_predicha.item():.2f}')  # se usa .2f para mostrar 2 decimales.
