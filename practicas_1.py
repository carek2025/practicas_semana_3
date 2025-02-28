import numpy as np
#------------------Practica 1------------------
#---------------Operaciones con Arrays:
#-----------------Objetivo: Realizar operaciones aritméticas con arrays y aplicar funciones de agregación.
#-------------------Ejercicio:
#---------------------Crea dos arrays A y B de tamaño (3,3) con valores aleatorios entre 1 y 10.
#---------------------Suma, resta, multiplica y divide los arrays entre sí.
#---------------------Calcula la media, varianza y desviación estándar de A y B.
#---------------------Encuentra el valor máximo y mínimo de cada array.

A=np.random.randint(1,11,(3,3))
B=np.random.randint(1,11,(3,3))

print(A)
print("")

print(B)
print("")

print(f"plus= {A+B}, minus= {A-B}, multiplication= {A*B}, Division= {A/B}, media-A - B ={A.mean()} - {B.mean()}, variancia-A - B= {np.var(A)} - {B.var()}, desviacion estandar-A - B: {A.std()} {B.std()}, valor maximo-minimo de A: {A.max(), B.min()}, valor maximo-minimo: {B.max(),B.min()}")

#-------------------Practica 2-------------------------
#-------------Indexación y Slicing Avanzado:
#---------------Objetivo: Manipular y extraer datos de un array usando indexación avanzada.
#-----------------Ejercicio:
#-------------------Crea un array M de tamaño (5,5) con números enteros aleatorios del 1 al 50.
#-------------------Extrae la tercera columna de M.
#-------------------Extrae la fila 2 a la 4 de M.
#-------------------Extrae los elementos que sean mayores a 20.
M=np.random.randint(1,51,(5,5))
print(M)

N=[i[:][2] for i in M]
print(N)

I=M[2:5]
print(I)

J=M[M[:]>20]
print(J)
#---------------------Practica 3--------------------------
#---------------Algebra Lineal con NumPy:
#-----------------Objetivo: Aplicar operaciones de álgebra lineal con matrices.
#-------------------Ejercicio:
#---------------------Crea una matriz cuadrada X de tamaño (4,4) con valores aleatorios entre 1 y 10.
#---------------------Calcula su determinante.
#---------------------Obtén la matriz inversa de X.
#---------------------Realiza el producto matricial de X por otra matriz aleatoria (4,4).
print("")
X=np.random.randint(1,11,(4,4))
print(X)

x_deter=np.linalg.det(X)
print(x_deter)

x_inversa=np.linalg.inv(X)
print(x_inversa)

Y=np.random.randint(1,11,(4,4))
print(np.linalg.multi_dot([X,Y]))
print(np.dot(X,Y))

#------------Practica 4-------------
#------------Generación de Datos y Distribuciones:
#--------------Objetivo: Generar datos siguiendo distribuciones estadísticas.
#----------------Ejercicio:
#------------------Genera un array de 1000 valores siguiendo una distribución normal con media 50 y desviación estándar 15.
#------------------Genera una distribución uniforme de 500 valores entre 0 y 1.
#------------------Calcula la media y la desviación estándar de cada distribución

array=np.random.normal(loc=50,scale=15,size=1000)
print(array)

array2=np.random.uniform(0,1,500)
print("")
print(array2)

print(f"Media de Array: {array.mean()}")
print(f"Media de Array2: {array2.mean()}")

print(f"Desviacion Estandar de array1: {array.std()}")
print(f"Desviacion Estandar de array2: {array2.std()}")

import numpy as np
#-------------Practica 5---------------------
#-----------------Trabajando con Datos Faltantes:
#--------------------Objetivo: Manejar valores NaN en un array.
#----------------------Ejercicio:
#------------------------Crea un array D de tamaño (6,6) con valores aleatorios del 1 al 100.
#------------------------Introduce valores NaN en posiciones aleatorias.
#------------------------Encuentra las posiciones donde hay NaN.
#------------------------Reemplaza los NaN con la media de la matriz.

D=np.random.randint(1,101,(6,6)).astype(float)

num_porsiciones_a_cambiar=np.random.randint(1,D.size)
posiciones=np.random.choice(D.size,num_porsiciones_a_cambiar,replace=False)
for pos in posiciones:
    fila,columna=np.divmod(pos,D.shape[1])
    D[fila][columna]=np.nan
print(D)

nan_positions=[(indice1,indice2) for indice1,fila in enumerate(D) for indice2,valor in enumerate(fila) if np.isnan(valor)]


for i1,i2 in nan_positions:
    D[i1][i2]=np.random.randint(100,201)
print("")
print(D)

#--------------------practica 6------------
#-----------------Aplicación de Funciones a Arrays
#-------------------Objetivo: Aplicar funciones personalizadas sobre arrays.
#---------------------Ejercicio:
#-----------------------Crea un array Z de tamaño (10,10) con valores aleatorios entre -50 y 50.
#-----------------------Escribe una función que reemplace los valores negativos por 0 y aplícala a Z.
#-----------------------Escribe otra función que normalice los valores de Z entre 0 y 1.

Z=np.random.randint(-50,50,(10,10))

print("")
print(Z)
print("")

#1era forma
def verificar_matriz(matriz):
    for i in range(0,(matriz.shape[0])):
        for j in range(0,(matriz[i].shape[0])):
            if matriz[i][j]<0:
                matriz[i][j]=0
    return matriz
new_z2=verificar_matriz(Z)
print(new_z2)
#2da forma
def change_negative_elements_to_zero(x):
    if x<0:
        x=0
        return x
    else:
        x=x
        return x
new_z1=np.vectorize(change_negative_elements_to_zero)(Z)
print(new_z1)        


def normalize_matriz(matriz):
    #Formula general para normalizar una matriz a 1
    matriz_normalizada=((matriz-matriz.min())/(matriz.max()-matriz.min()))
    return matriz_normalizada
print(normalize_matriz(Z))
