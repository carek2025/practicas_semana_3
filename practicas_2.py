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

#----------------------------Practica 7------------------
w=np.random.randint(10,101,(3,3))
