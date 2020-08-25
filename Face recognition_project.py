#Face Recognition project Using Concept OF ML
import face_recognition as fr
import cv2
import numpy as np
import os
def load_images_from_folder(folder):  #folder is a parameter which will store the name of the folder
    images=[]
    for filename in os.listdir(folder):      #os.listdr() will create a list of all the the files in the folder
        images.append(filename)
    return images

video_capture=cv2.VideoCapture(0)
images_name=load_images_from_folder('images')    #this will store the names of images in the list
#print(images)
images=[]
for img in images_name:
    images.append(fr.load_image_file(os.path.join("images",img)))  #this will load the complete image in the list
encodings=[]
for img in images:
    encodings.append( fr.face_encodings(img)[0] )
    
#creating array of known Face encodings and there name
known_face_encodings=[]
for encode in encodings:
    known_face_encodings.append(encode)
known_face_names=[]
for name in images_name:
    known_face_names.append(os.path.splitext(name)[0])

#initialize some variables (these are basically for group photo)
face_locations=[]
face_encodings=[]
face_names=[]
process_this_frame=True
while True:
    #grab a single frame of vedio
    ret,frame=video_capture.read()
    #Resize the frame of vedio to 1/4th size for fast processing
    small_frame=cv2.resize(frame,(0,0), fx=0.25 , fy=0.25)
    #convert the image from BGR color(used by openCv) to RGB color (used by face recognition)
    rgb_small_frame=small_frame[:,:,::-1]
    #only process every other frame of vedio to save time
    if process_this_frame:    #this variable is initialised to True above
        #find all the face and face encoding in the current frame of vedio
        face_locations=fr.face_locations(rgb_small_frame)
        face_encodings=fr.face_encodings(rgb_small_frame, face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            #see if the face is known or not
            matches=fr.compare_faces(known_face_encodings,face_encoding)
            name="Unknown"
            face_distances=fr.face_distance(known_face_encodings,face_encoding)
            best_match_index=np.argmin(face_distances)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]
            face_names.append(name)
    process_this_frame=not process_this_frame
    #display the result
    for (top,right,bottom,left),name in zip(face_locations,face_names):
        top*=4 #as earlier we reduced the frame by 25% therefore
        right*=4
        bottom*=4
        left*=4
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,225),2)
        cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,225),cv2.FILLED)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left+6, bottom-6),font,1.0,(225,225,225),1)
    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

        
    
    






    
    
