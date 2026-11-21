import os
import urllib.request

import cv2


DATA_FOLDER = os.path.join(os.path.dirname(__file__), "faces")
NUMBER_OF_PICTURES = 20
MATCH_THRESHOLD = 0.45


def get_face_cascade_path():
    candidates = [
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascades"),
        os.path.join(os.path.dirname(cv2.__file__), "data"),
        os.path.join(os.path.dirname(__file__), "data", "haarcascades"),
    ]

    for folder in candidates:
        file_path = os.path.join(folder, "haarcascade_frontalface_default.xml")
        if os.path.exists(file_path):
            return file_path

    download_folder = os.path.join(os.path.dirname(__file__), "data", "haarcascades")
    os.makedirs(download_folder, exist_ok=True)
    file_path = os.path.join(download_folder, "haarcascade_frontalface_default.xml")

    url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
        "haarcascade_frontalface_default.xml"
    )
    urllib.request.urlretrieve(url, file_path)
    return file_path


face_detector = cv2.CascadeClassifier(get_face_cascade_path())


def get_face(gray, x, y, width, height):
    face = gray[y:y + height, x:x + width]
    face = cv2.resize(face, (150, 150))
    face = cv2.equalizeHist(face)
    return face


def enroll_face():
    name = input("Enter the person's name: ").strip()
    name = "".join(c for c in name if c.isalnum() or c in " _-")

    if name == "":
        print("Invalid name.")
        return

    person_folder = os.path.join(DATA_FOLDER, name)
    os.makedirs(person_folder, exist_ok=True)

    camera = cv2.VideoCapture(0)
    count = 0

    print("Look at the camera and slowly move your head.")
    print("Press Q to stop.")

    while count < NUMBER_OF_PICTURES:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # Save a picture only when one face is visible.
        if len(faces) == 1:
            x, y, width, height = faces[0]
            face = get_face(gray, x, y, width, height)

            count += 1
            filename = os.path.join(person_folder, str(count) + ".jpg")
            cv2.imwrite(filename, face)

            cv2.rectangle(
                frame, (x, y), (x + width, y + height), (0, 255, 0), 2
            )

        cv2.putText(
            frame,
            "Pictures: " + str(count) + "/" + str(NUMBER_OF_PICTURES),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Enroll Face", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Saved " + str(count) + " pictures for " + name + ".")


def load_faces():
    saved_faces = []

    if not os.path.exists(DATA_FOLDER):
        return saved_faces

    for name in os.listdir(DATA_FOLDER):
        folder = os.path.join(DATA_FOLDER, name)

        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                image = cv2.imread(
                    os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE
                )
                if image is not None:
                    saved_faces.append((name, image))

    return saved_faces


def compare_face(face, saved_faces):
    best_name = "Unknown"
    best_score = -1

    for name, saved_face in saved_faces:
        # This gives a similarity score from -1 to 1.
        score = cv2.matchTemplate(
            face, saved_face, cv2.TM_CCOEFF_NORMED
        )[0][0]

        if score > best_score:
            best_score = score
            best_name = name

    # The rule: a low score means the face is not recognized.
    if best_score < MATCH_THRESHOLD:
        best_name = "Unknown"

    return best_name, best_score


def recognize_faces():
    saved_faces = load_faces()

    if len(saved_faces) == 0:
        print("No saved faces. Enroll a person first.")
        return

    camera = cv2.VideoCapture(0)
    print("Press Q to stop.")

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for x, y, width, height in faces:
            face = get_face(gray, x, y, width, height)
            name, score = compare_face(face, saved_faces)

            color = (0, 255, 0)
            if name == "Unknown":
                color = (0, 0, 255)

            cv2.rectangle(
                frame, (x, y), (x + width, y + height), color, 2
            )
            cv2.putText(
                frame,
                name + " " + str(round(score, 2)),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1 - Enroll a face")
    print("2 - Recognize faces")
    choice = input("Choose 1 or 2: ")

    if choice == "1":
        enroll_face()
    elif choice == "2":
        recognize_faces()
    else:
        print("Invalid choice.")