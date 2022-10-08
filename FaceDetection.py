import cv2


cap = cv2.VideoCapture(0)


human_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    i=0

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.9, 1)
    
    # Display the resulting frame
    for (x,y,w,h) in humans:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    i = i+1
 
        # Adding face number to the box detecting faces
    cv2.putText(frame, 'Face Number:'+str(i), (x-10, y-10),
             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(frame, i)
 
    
         
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
