import cv2


# Create our body classifier
body_classifier = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/cv2/data/haarcascade_fullbody.xml')

# Initiate video capture for video file
vid = cv2.VideoCapture('/Users/apple/Downloads/PRO-106-ProjectTemplate-main/walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = vid.read()

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2,3)
    print(len(bodies))
    # Extract bounding boxes for any bodies identified
    for(x,y,w,h)in bodies:
        cv2.rectangle(vid,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = vid[y:y+h, x:x+w]
        cv2.imwrite("bodies.avi",roi_color)

    cv2.imshow("Web cam",frame)
    cv2.waitKey(0)
    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

vid.release()
cv2.destroyAllWindows()