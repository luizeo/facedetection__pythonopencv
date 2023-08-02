import cv2
import numpy as np
video = cv2.VideoCapture(0)
classificadorFace = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')
amostra = 1
numAmostras = 30
id = input("Digite um identificador: ")
largura,altura = 220, 220
print("Capturando as faces...")
while True:
    conectado, frame = video.read()
    #print(conectado)
    #print(frame)
    imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(np.average(imagemCinza))
    facesDetectadas = classificadorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150,150))
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = frame[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho)
        print(olhosDetectados)
        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagemCinza) > 110:  #Média dos valores RGB
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                    cv2.imwrite("fotos/pessoas." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                    print("[foto " + str(amostra) + " capturada com sucesso]")
                    amostra += 1
    cv2.imshow('Face', frame)
    cv2.waitKey(1)
    if (amostra > numAmostras):
        break

print("Faces capturadas com sucesso.")
video.release()
cv2.destroyAllWindows()