import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
#Imagen
img = mpimg.imread('C:/Claseses python/pythoncode/foto.jpg')
#Normalizar la imagen como el valor de cada pixel de 0 a 255 
# - segun la presencia de los colores en rango de 0 a 1
img = np.array(img, dtype=np.float64)/255
#Obtenemos  las dimensiones de la imagen
# Ancho, Alto y profundidad, el numero de canales son 3 (RGB)
h,w,d= img.shape
n_classes = 8
# Reshape para obtener un vector de los valores de cada canal
img_array = np.reshape(img, (w * h,d))
#Revolvemos los datos y tomamos los primeros 1000 valores
image_array_sample = shuffle(img_array, random_state=0)[:1000]
#Creamos nuestro modelo para el Kmeans con 8 clusters 
kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(image_array_sample)
#imprimir los centroides por cada cluster
print("Centroides por cada canal")
print(kmeans.cluster_centers_)
#Prediccion de la imagen completa
label = kmeans.predict(img_array)
img_labels_per_chanel = np.reshape(label,(h,w))
#Creamos una matriz de ceros para guardae el valor de l centroide
img_out= np.zeros((h,w,d))
for i in range(h):
    for j in range(w):
        img_out[i][j][0] = kmeans.cluster_centers_[img_labels_per_chanel[i][j]][0]
        img_out[i][j][1] = kmeans.cluster_centers_[img_labels_per_chanel[i][j]][1]
        img_out[i][j][2] = kmeans.cluster_centers_[img_labels_per_chanel[i][j]][2]

#Tambien se pude hacer que solo 8 centreides para todo los canales 
img_array = np.reshape(img,(w * h * d, 1))
#resolvemos los datos y tomammos los primeros 1000 valores
image_array_sample = shuffle(img_array,random_state=0)[:1000]
kmeans2= KMeans(n_clusters= n_classes, random_state=0).fit(image_array_sample)

#Imprimimos los centroides de cada cluster
print("\nCentroides para todos los canales")
print(kmeans2.cluster_centers_)

#Hacemos la predicion de la imagen completa
labels= kmeans2.predict(img_array)

img_out_wo_channels = np.zeros((h, w, d))
for i in range(h):
    for j in range(w):
        for k in range(d):
            img_out_wo_channels[i][j][k]= kmeans2.cluster_centers_[labels[i*w*d + j*d + k]]

plt.figure()
plt.title('Original image')
plt.imshow(img)
plt.show()
plt.figure()
plt.title('Quantized image witg K-Means 8 clusters per channel')
plt.imshow(img_out)
plt.show()
plt.figure()
plt.title('Quantized image with K−Means 8 clusters for all channel')
plt.imshow(img_out_wo_channels)
plt.show()

print("Clase para el pixel en la fila 374, columna 577")
print(img_labels_per_chanel[374][577])
print("Valor para cada canal")
print(kmeans.cluster_centers_[img_labels_per_chanel[374][577]])
print("Clase y valor para el pixel en la fila 374, columna 577 en el canal R")
print(labels[374*w*d+577*d+0],kmeans2.cluster_centers_[labels[374*w*d+577*d+0]])
print("Clase y valor para el pixel en la fila 374, columna 577 en el canal G")
print(labels[374*w*d+577*d+1],kmeans2.cluster_centers_[labels[374*w*d+577*d+1]])
print("Clase y valor para el pixel en la fila 374, columna 577 en el canal B")
print(labels[374*w*d+577*d+2],kmeans2.cluster_centers_[labels[374*w*d+577*d+2]])

