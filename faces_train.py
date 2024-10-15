import os
import cv2 as cv
import numpy as np

people=['Narendra_Modi','Barack_Obama','Swami_Vivekananda','Bill_Gates','Elon_Musk']

# p=[]
# for i in os.listdir(r'C:\Users\indre\Desktop\Books\coding\OpenCV\opencvlecture\Faces'):
#     p.append(i)

# print(p)

DIR=r'C:\Users\indre\Desktop\Books\coding\OpenCV\opencvlecture\Faces'
haar_cascade=cv.CascadeClassifier('haar_face.xml')

features=[]
labels=[]
def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for(x,y,w,h) in faces_rect:
                faces_roi=gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ------------')
# print(f'Length of the features={len(features)}')
# print(f'Length of the labels={len(labels)}')

features=np.array(features,dtype='object')
labels=np.array(labels)

face_recognizer=cv.face.LBPHFaceRecognizer_create()

#Train tne Recognizer on the features list and labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy',features)
np.save('labels.npy',labels)