import cv2
import numpy as np
import os

def leer_imagen(nombre_imagen):
    return cv2.imread("input/" + str(nombre_imagen))

def convertir_imagenes_escala_grises(imagen):
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

def dilatar_mascara_get_canny(imagen):
    canny = cv2.Canny(imagen, 10, 150)
    return cv2.dilate(canny, None, iterations=3)

def guardar_imagen(nombre_imagen, imagen):
    cv2.imwrite('output/' + nombre_imagen, imagen)
    
def guardar_imagen_dilatada(nombre_imagen, imagen):
    cv2.imwrite('dilated/' + nombre_imagen, imagen)
    
def guardar_imagen_contornos(nombre_imagen, imagen, contornos):
    tmp = imagen.copy()
    cv2.imwrite('contornos/' + nombre_imagen, cv2.drawContours(tmp, contornos, -1, (0,255,0), 3))

for imagen in os.listdir('input'):
    img = leer_imagen(imagen)
    imagen_escala_grises = convertir_imagenes_escala_grises(img)
    canny = dilatar_mascara_get_canny(imagen_escala_grises)
    
    guardar_imagen_dilatada(imagen, canny)
    
    cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    guardar_imagen_contornos(imagen, img, cnts)
    
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > img.shape[1] * 0.25:
            # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            temp = img[y:y+h, x:x+w]
            temp = convertir_imagenes_escala_grises(temp)
            temp = cv2.threshold(temp, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            print(np.sum(temp != 0))
            print(temp.shape[0] * temp.shape[1])
            """ cv2.imshow("temp", temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows() """
            if int((np.sum(temp != 0) / (temp.shape[0] * temp.shape[1]) * 100)) > 15:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            else:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            """ print(temp.shape[2])
            print("Total Pixeles: " + str(np.sum(temp)))
            print("Pixeles Blancos: " + str(np.sum(temp != 0)))
            print("Division: " + str(int((np.sum(temp != 0) / (temp.shape[0] * temp.shape[1]) * 100)))) """
        else:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    guardar_imagen(imagen, img)