#from sklearn-->es una abreviatura de scikit-learn que contiene herramientas(packetes y modulos) para hacer machine learning
from sklearn import datasets     #Se usa para cargar datasetd establecidos que son conjuntos de datos son como tablas tiene la caracteristica o sus datos o medidas y sus etiquetas o nombre de la columna,son como los dataframe pero tiene otras caracteristicas   
from sklearn.model_selection import train_test_split #Se usa para dividir los datos en dos partes ;uno para entrenar el modelo y otra para probarlo
#svm --> es un modulo que se usa para trabajar con maquinas de soporte vectorial(SVM) que son un tipo de modelo de machine learning que se utiliza principalmente en clasificacion(poner datos en sus etiquetas o categorias) y regresion(predecir un valor numerico) 
from sklearn.svm import SVC #svc-->Support Vector Classifier -->Es un modelo de svm mas comunmente utilizado para clasificar datos en categorias
from sklearn.metrics import accuracy_score #Para medir la precision del modelo o que tan bien funciona el modelo

#Cargamosx el dataset de iris(contiene medidas de flores de iris de tres especies y se utiliza para predecir la especie a partir de las caracteristicas)
iris=datasets.load_iris()
x=iris.data # saca las caracteristicas o datos que tiene el dataset que este caso es de flores(en este caso de iris es largo y ancho las caracteristicas)
y=iris.target  #saca las etiquetas es decir el tipo de flor (en este caso seria iris-setosa, iris-versicolor, iris-virginica)

#Desempaquetamos lo que nos entrega la funcion,la funcion tiene 4 argumentos el primero es la variable donde se almacenaron las caraccteristicas ,el segundo es la variable donde se almacenaron las etiquetas,el tercero que es un argumento nombrado que tanto por cienta(%) de los datos son para prueba y el resto se pone para entrenar de forma automatica,el cuarto que es la aleatorizacion si no lo pones o lo defines con un numero cualquiera siempre te dara diferentes divisiones de datos o conjuntos de entrenamoiento y prueba cada ves que ejecutes tu codigo, ya que cada vez se inicializa con un numero aleatorio distinto lo que te puede llevar a resultados inconsistentes si no lo defines eso dificulta la comparacion de modelos de manera justa
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#creamos el modelo de clasificacion con un kerner linear es decir que el modelo va a intentar separar las clases con una linea recta,el kernel puede tomar los siguientes valores: poly(polinomico se utiliza si quieres nun espacio de mayor dimension,lo que permite separa los datos de no lineal,si pones esto tienes que agregar otro parametro con un argumento nombrado que es el degree que controla el grado del polinopmio osea seria asi K(x,y)=(x*y+c) elebado a la 3 o cubo,donde el x y y son los puntoss donde se evalua el kernel y c es el parametro de desplazamiento,dependiendo del degree se producen modelos con mas o menos capacidad para capturar relaciones complejas en los datos un modelo con mayor degree es mas flexibles osea que puede ajustarse mejor a los datos coplejos,pero tambien puede resultar en un ajuste excesivo(overfitting) que es cuando el modelo aprende demasiado bien los datos de entrenamiento y no generaliza bien a nuevos datos,resumienod un pequeño degree seria para datos sensillos y un degree mayor puede ayudar a capturar relaaciones mas complejas)
model = SVC(kernel='linear')

# metodo que nos permite que entrenamos el modelo que fue creado anteriormente con los datos de entrenamiento--x_train serian las caracteristicas y y_train seria las etiquetas que desempacamops anteriormente
model.fit(x_train, y_train)

# Una vez que el modelo fue entrenado utilizamos el metodo y predecimos(o intenta adivinar) el tipo de flor para los datos de prueba
y_pred = model.predict(x_test) #-->caracteristicas de prueba
 
# metodo que nos permite comparar las predicciones con las etiquetas reales osea lo compara con las respuestas correctas pra calcular la precision del modelo(cuantas veces el modelo adivino correctamente)
accuracy = accuracy_score(y_test, y_pred)#el ocurracy o la precion del modelo
#Como la precision generalmente se da en un rango de [0,1],al multiplicarlo por 100 lo convierte en un valor de porcentaje,el : indica el inicio de la especificacion de formato;
#el .2f indica formatear como numero decimal(f) con 2 decimales(.2),se redondea automaticamente,si le pones .0f entondes es con 0 decimales a esto se le conoce formatos para numeros decimales
#Hay otras opciones que te da el f string que son formatos para enteros y alineacion-->:d(numero entero),:5d(alineacion con 5 espacios,se mando al final),:<5d(alineacion a la izquierda),:^5d(alineacion al centro)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')