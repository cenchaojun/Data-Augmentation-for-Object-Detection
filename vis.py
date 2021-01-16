import cv2
import xml.etree.ElementTree as ET
import os
xml_path = './data/cloudxml'
file_path = './data/cloud'

for image in os.listdir(file_path):
    image_prefix, image_suffix = image.split('.')
    xml_file = image_prefix + '.xml'
    image_file_path = os.path.join(file_path,image)
    # xml_list = [xml for xml in os.listdir(xml_path)]
    # if xml_file in xml_list:
    #     print("yes")
    xml_file_path = os.path.join(xml_path,xml_file)
    print(xml_file_path)
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # 读图像一定要在for循环之前读，要不然重新读，就写不进去框了
    img = cv2.imread(image_file_path)
    for bndbox in root.findall('./object/bndbox'):
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(255, 0, 0), thickness=3)
    cv2.imwrite('./data/cloudvis/{}'.format(image), img)
