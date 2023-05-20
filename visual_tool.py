
import os, cv2
import numpy as np
import xml.etree.ElementTree as ET
from tensorflow import keras

X, y = [], []
model = keras.models.load_model("smoke_detection.h5")

def get_image_bounding_box(xml_path):
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bb = root.findall("./object/bndbox")[0]

    xmin = int(bb.find("xmin").text)
    xmax = int(bb.find("xmax").text)
    ymin = int(bb.find("ymin").text)
    ymax = int(bb.find("ymax").text)
    
    return ((xmin, ymin), (xmax, ymax))

def get_model_pred(x):
    global model
    pred = model.predict(np.array([cv2.resize(x, (640, 480))]))[0]
    xmin, ymin, xmax, ymax = list(map(int, pred))
    print("Model Pred -", ((xmin, ymin), (xmax, ymax)), "\n")
    return ((xmin, ymin), (xmax, ymax))

def map_image_xml(path):
    use_model = True
    for i in os.listdir(path + "/images"):
        fname = i.replace(".jpg", "")
        print("Reading", fname)
        imgg = cv2.imread(path + "/images" + "/" + i)
        print(imgg.shape)
        if use_model:
            print("XML -", get_image_bounding_box(path + "/annotations/" + fname + ".xml"))
            img = cv2.rectangle(imgg, *get_model_pred(imgg), (255, 0, 0), 2)
            img = cv2.rectangle(img, *get_image_bounding_box(path + "/annotations/" + fname + ".xml"), (0, 255, 0), 2)
        else:
            img = cv2.rectangle(img, *get_image_bounding_box(path + "/annotations/" + fname + ".xml"), (255, 0, 0), 2)
        #start_point, end_point, color, thickness
        cv2.imshow(fname, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
def map_from_custom_set(path):
    for i in os.listdir(path):
        print("Reading", i)
        try:
            imgg = cv2.resize(cv2.imread(path + "/" + i), (640, 480))
        except Exception as e:
            print("Error", e, "\n")
            continue
        img = cv2.rectangle(imgg, *get_model_pred(imgg), (255, 0, 0), 2)
        cv2.imshow(i, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#get_image_bounding_box(r"C:\Users\habee\Container\Projects\pythonProgs\SBNPD\archive\annotations\ck0k9aqm99o2o0721aml8qpqr_jpeg.rf.028c79a871f1964bd02ab8c4b5693e6d.xml")

map_image_xml("archive")

#map_from_custom_set("archive/Human_Test")