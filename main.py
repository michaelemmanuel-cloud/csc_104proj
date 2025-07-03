
import cv2
import pandas as pd


image_path = 'Screenshot 2025-07-03 003535.png'  
cascade_path = 'haarcascade_frontalface_default.xml'  


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path
    if 'haarcascade' in cascade_path else cascade_path)

# Load image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not load image: {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Store results
face_data = []

# Draw rectangles around the faces
for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face_data.append({
        'image': (image_path),
        'face_id': i + 1,
        'x': x,
        'y': y,
        'width': w,
        'height': h
    })

#result
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



df = pd.DataFrame(face_data)
df.to_csv('detected_faces.csv', index=False)

print(f"✅ Done! {len(faces)} face(s) detected and saved to detected_faces.csv")
