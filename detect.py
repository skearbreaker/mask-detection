# Gerekli kütüphaneleri projemize dahil edelim.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# Bir fotoğraf karesindeki yüzleri ve maskeleri tespit etmek için bir fonksiyon tanımlayalım.
def detector(frame, networkFace, networkMask):
    # Mevcut fotoğraf karenin boyutlarını alalım.
    (h, w) = frame.shape[:2]

    # Kareyi sayısal olarak işlemek için bir BLOB'a (Binary Large OBject) dönüştürelim.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # BLOB'u, yüzleri tespit etmek için yüz tespit ağından geçirelim.
    networkFace.setInput(blob)
    detections = networkFace.forward()

    # Bulunan yüzleri konsola basalım.
    print(detections.shape)

    # Yüzler, yüzlerin bulunduğu konumlar ve face mask network'ten elde edeceğimiz veriler için listeleri tanımlayalı.
    faces = []
    locations = []
    predictions = []

    # faceNet kullanarak tespit ettiğimiz sonuçlar üzerinde döngü başlatalım.
    for i in range(0, detections.shape[2]):
        # Tespitlerden güvenilir bulduğumuz sonuçları ayıklayalım.
        confidence = detections[0, 0, i, 2]

        # Belirlediğimiz minimum güvenilirlik seviyesi altında kalan sonuçları eleyelim.
        if confidence > 0.5:
            # Tespit edilen bölgenin alanını (x, y) koordinatlarına dönüştürelim.
            location = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = location.astype("int")

            # Tespit edilen bölgenin, fotoğraf karemizin sınırları dahilinde olduğundan emin olalım.
            (x1, y1) = (max(0, x1), max(0, y1))
            (x2, y2) = (min(w - 1, x2), min(h - 1, y2))

            # İlgilendiğimiz bölgedeki yüzü ayıklayalım.
            face = frame[y1:y2, x1:x2]

            # Ayıklanan yüzü, BGR renk formatından RGB renk formatına çevirelim.
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Fotoğrafı rahat işlemek için 224x224 boyutuna yeniden ölçeklendirelim.
            face = cv2.resize(face, (224, 224))

            # Yüzü işlemek için bir NumPY dizisine dönüştürelim.
            face = img_to_array(face)

            # Resmi işleyelim.
            face = preprocess_input(face)

            # Tespit ettiğimiz yüzü ve bölgeyi, ilgili listelerimize ekleyelim.
            faces.append(face)
            locations.append((x1, y1, x2, y2))

    # Maske tespit işelimimizi, karede en az bir yüz varsa gerçekleştirelim.
    if len(faces) > 0:
        # Operasyon hızını arttırmak için, yüzleri bir bir işlemektense, karede bulunan tüm yüzlerde toplu olarak
        # aynı anda maske bulmayı deneyeceğiz.
        faces = np.array(faces, dtype="float32")
        predictions = networkMask.predict(faces, batch_size=32)

    # Geriye, yüzlerin konumlarını ve bu konumlara ait tahminleri bir iki boyutlu bir vektör olarak döndürelim.
    return (locations, predictions)


# Yüz tespit modelimizi diskten yükleyelim.
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"

networkFace = cv2.dnn.readNet(prototxtPath, weightsPath)

# Daha önceden hazırladığımız maske tespit modelimizi diskten yükleyelim.
networkMask = load_model("mask_detector.model")

# Video akışını başlatalım.
print("[BİLGİ] Video akışı başlıyor...")
vs = VideoStream(src=0).start()

# Vieo akışından gelen tüm kareleri yakalamak için döngü başlatalım.
while True:
    # Video akışından ilgili kareyi alalım ve maksimum 400 miksel olacak şekilde limitleyelim.
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not

    # Video karesindeki yüzleri, ve yüzlerin konumundaki maske tahminlerini işleyelim.
    (locations, predictions) = detector(
        frame, networkFace, networkMask)

    # Bulunan yüzler ve bu yüzlerin konumları üstünde bir döngü başlatalım.
    for (location, prediction) in zip(locations, predictions):
        # Bölge ve tahminleri ayıklayalım.
        (x1, y1, x2, y2) = location
        (mask, noMask) = prediction

        # Tahmine göre bir mesaj belirleyelim.
        message = "Maskeli" if mask > noMask else "Maskesiz"

        # Tahmine göre renk atayalım, maskeli için yeşil, maskesiz için kırmızı.
        color = (0, 255, 0) if mask > noMask else (0, 0, 255)

        # Maske olasılığını mesaja dahil edelim.
        message = "{}: {:.2f}%".format(message, max(mask, noMask) * 100)

        # Mesajı bölgenin içine yazalım.
        cv2.putText(frame, message, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Bölgeyi kare üzerine çizelim.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Kareyi ekrana çizdirelim.
    cv2.imshow("Frame", frame)

    # Programı sonlandırmak için bir tuş atayalım.
    key = cv2.waitKey(1) & 0xFF

    # Eğer kullanıcının bastığı tuş, atadığımız tuş ise programı sonlandıralım.
    if key == ord("q"):
        break

# Program için işgal ettiğimiz sistem kaynaklarını serbest bırakalım.
cv2.destroyAllWindows()
vs.stop()
