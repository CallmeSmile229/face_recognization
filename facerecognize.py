import face_recognition
import cv2
import numpy as np
import imutils
import requests

url = "http://192.168.1.10:8080/shot.jpg"

TranKimChi_image = face_recognition.load_image_file("Kiet.jpg")
TranKimChi_face_encoding = face_recognition.face_encodings(TranKimChi_image)[0]

my_image = face_recognition.load_image_file("me.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]

known_face_encodings = [
    TranKimChi_face_encoding,
    my_face_encoding
]
known_face_names = [
    "Kiet",
    "Me"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Using IP webcam to connect with camera in your phone to improve quality of video.
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    frame = imutils.resize(img, width=1000, height=1800)

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cv2.destroyAllWindows()