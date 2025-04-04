import easyocr
import cv2

haarcascade = "haarcascade_russian_plate_number.xml"

video_path = 'test2.mp4'
cap = cv2.VideoCapture(video_path)
reader = easyocr.Reader(['en'])
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    succes, img = cap.read()
    if not succes:
        break

    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    plate_cascade = cv2.CascadeClassifier(haarcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (39,96,230), 2)

        img_plate = img[y:y + h, x:x + w]

        img_plate_resized = cv2.resize(img_plate, (w * 2, h * 2))

        if x + img_plate_resized.shape[1] > img.shape[1]:
            new_width = img.shape[1] - x
            img_plate_resized = cv2.resize(img_plate_resized, (new_width, img_plate_resized.shape[0]))

        if y - img_plate_resized.shape[0] >= 0:
            img[y - img_plate_resized.shape[0]:y, x:x + img_plate_resized.shape[1]] = img_plate_resized
        else:
            height_available = y
            if height_available > 0:
                if img_plate_resized.shape[0] > height_available:
                    aspect_ratio = img_plate_resized.shape[1] / img_plate_resized.shape[0]
                    new_height = height_available
                    new_width = int(new_height * aspect_ratio)
                    img_plate_resized = cv2.resize(img_plate_resized, (new_width, new_height))

                # Place it at the top of the image
                img[0:img_plate_resized.shape[0], x:x + img_plate_resized.shape[1]] = img_plate_resized

        results = reader.readtext(img_plate_resized)

        for (bbox, text, prob) in results:
            cv2.putText(img, f'Plate: {text}', (x, y - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (145,230,9), 1, cv2.LINE_AA)

    cv2.imshow("Car Plate", img)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()