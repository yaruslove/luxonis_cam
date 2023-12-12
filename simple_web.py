import cv2
import os


path_imgs = "/Users/YaVolkonskiy/Documents/001_Projects/003_tomato/003_hardware/luxonis/saved_photo/"
count_image_save = 24


# Initialize the VideoCapture object to use the default camera (camera index 0 is webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to continuously get frames from the camera
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow('Video Feed', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('p'):
        update_path = os.path.join(path_imgs, f"{count_image_save}.png")
        cv2.imwrite(update_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        count_image_save +=1 

cap.release()
cv2.destroyAllWindows()