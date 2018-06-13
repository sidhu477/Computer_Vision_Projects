# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections

def ob_det(image, model, transform):
    height, width = image.shape[:2]
    image_t = transform(image)[0]
    a = torch.from_numpy(image_t).permute(2,0,1)
    a = Variable(a.unsqueeze(0))
    b = model(a)
    img_det = b.data
    normalizing_co_ordinates = torch.Tensor([width,height,width,height]) 
    # img_det = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]    
    for i in range (img_det.size(1)):
        j = 0
        while img_det[0,i,j,0] >=0.6:
            point = (img_det[0,i,j,1:]*normalizing_co_ordinates).numpy()
            cv2.rectangle(image, (int(point[0]),int(point[1])),(int(point[2]),int(point[3])),(255,0,0),2)
            cv2.putText(image, labelmap[i-1],(int(point[0]),int(point[1])),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            j+=1
            
    return image

# Creating the SSD neural network
model = build_ssd('test')
model.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

# Creating the transformation
img_transformation = BaseTransform(model.size, (104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('op.mp4', fps = fps)
for i, image in enumerate(reader):
    image = ob_det(image, model.eval(), img_transformation)
    writer.append_data(image)
    print(i)
writer.close()    
    





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    