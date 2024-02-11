import pandas as pd
import numpy as np
# import bbox_visualizer as bbv
import os
import glob
from moviepy.editor import VideoFileClip
from random import choice
# import cv2
from PIL import ImageDraw, Image
from datetime import datetime
import pytz

def calculate_iou(bbox1, bbox2):
    '''calculates intersection over union
    
    PARAMETERS:
        bbox1: ultralytics.engine.results.Boxes
        bbox2: ultralytics.engine.results.Boxes
    RETURNS:
        float
    '''
    bbox1,bbox2 = bbox1.xyxy.numpy()[0],bbox2.xyxy.numpy()[0]
    w=0
    h=0
    if (bbox1[0] <= bbox2[2]) and (bbox2[0] <= bbox1[2]):
        w = np.min((bbox1[2],bbox2[2])) - np.max((bbox1[0],bbox2[0]))
        if (bbox1[1] <= bbox2[3]) and (bbox2[1] <= bbox1[3]):
            h = np.min((bbox1[3],bbox2[3])) - np.max((bbox1[1],bbox2[1]))
    overlap = w*h
    area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    iou = overlap/(area1+area2-overlap)
    return iou


def delete_overlaps(boxes,threshold):
    '''deletes bboxes with less confidance among bboxes with iou >= threshold

    PARAMETERS:
        boxes: ultralytics.engine.results.Boxes
        threshold: float [0,1]
    RETURNS:
        list of ultralytics.engine.results.Boxes
    '''
    stop = False
    bboxes = dict(enumerate(boxes))
    if len(bboxes) <2:
        stop = True
    i = 0
    j = 1
    while not stop:
        bbox1 = bboxes[i]
        bbox2 = bboxes[j]
        iou = calculate_iou(bbox1,bbox2)
        if iou >= threshold:
            # print(iou)
            # print(cl_map[bbox1.cls.numpy()[0]],bbox1.conf.numpy())
            # print(cl_map[bbox2.cls.numpy()[0]],bbox2.conf.numpy())
            useless_box_id = np.argmin((bbox1.conf.numpy()[0],bbox2.conf.numpy()[0]))
            # bboxes.pop(useless_box_id)
            if useless_box_id == 0:
                bboxes.pop(i)
                i+= 1
            else:
                bboxes.pop(j)
                j+= 1
        else:
            j+= 1
        if j > max(list(bboxes.keys())):
            i+= 1
            j = i+1
        if i >= sorted(list(bboxes.keys()))[-1]:
            stop = True
        # print(list(bboxes.values()))
    return list(bboxes.values())



def illustrate_boxes(preds,cl_map,threshold, img):
    '''draws bboxes 

    PARAMETERS:
        preds: ultralytics.engine.results.Results, predictions 
        cl_map: dict, class map
        threshold: float, threshold for iou
        img: PIL.Image, image to which add bboxes
    RETURNS:
        PIL.Image object 
    '''
    colors = [(190, 110, 70),(198, 171, 123),(205, 231, 176),(163, 191, 168),(139, 163, 164),(114, 134, 160),(11, 79, 108)]
    img1 = Image.fromarray(img)
    draw = ImageDraw.Draw(img1)
    boxes = delete_overlaps(preds.boxes,threshold) 
    labels = []
    bboxes = []
    #unpack bboxes and labels from yolo 
    for i in range(len(boxes)):
        bbox = boxes[i].xyxy[0].numpy()
        bbox = list(map(int,bbox))
        bboxes.append(bbox)
        label = cl_map[int(boxes[i].cls[0].numpy())] + '    ' + str(np.round(boxes[i].conf[0].numpy(),2))
        labels.append(label)
   
    
    used_labels = dict()
    for l in range(len(labels)):
        #same color for same class
        short_label = labels[l].split('    ')[0]
        if short_label in used_labels:
            clr = used_labels[short_label]
        else:
            clr = choice(colors)
            used_labels[short_label] = clr
            colors.remove(clr)
        
        bbox = bboxes[l]
        w_bbox = bbox[2]-bbox[0]
        bbox_middle = w_bbox//2 + bbox[0]
        #this way smol bboxes don't look ugly af
        if w_bbox <= 100:
            thickness = 1
            size = 8
        elif w_bbox <= 200:
            thickness = 2
            size = 12
        elif min(img.shape[:1]) <= 450:
            thickness = 4
            size = 30
        else:
            thickness = 5
            size = 35
        
        draw.rounded_rectangle(bbox,outline=clr,width=thickness,corners=(True,True,True,True),radius=4)
        w_text = int(draw.textlength(labels[l],font_size=size))
        #for label to stay within the bbox
        while w_bbox < w_text:
            size = size-1
            w_text = int(draw.textlength(labels[l],font_size=size))
        
        text_bbox = draw.textbbox((bbox_middle,bbox[1]),labels[l],anchor='mt',font_size=size)
        text_bbox = list(text_bbox)
        text_bbox[0:3:2] = bbox[0:3:2]
        text_bbox[3] += 2*thickness
        draw.rounded_rectangle(text_bbox,fill=clr,outline=clr,width=thickness,corners=(True,True,True,True),radius=4)
        draw.text((bbox_middle,bbox[1]+thickness),labels[l],anchor='mt',font_size = size, fill = (0,0,0))
        
        # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),clr,thickness)
        # (label_width, label_height), baseline = cv2.getTextSize(label, font, size, thickness)
        # label_bg = [bbox[0], bbox[1], bbox[0] + label_width+5, bbox[1] + label_height + int(14 * size)+ 0.5]
        # cv2.rectangle(img, (label_bg[0], label_bg[1]),(label_bg[2] + 5, label_bg[3]), clr, -1)
        # cv2.putText(img,label,(bbox[0] + 5, bbox[1] + int(16 * size) + (4 * thickness)), font, size, (0,0,0), thickness)

    # clr = choices(colors,k=len(labels))
    # clr = choice(colors)
    # img1 = bbv.draw_multiple_rectangles(img,bboxes,thickness=thickness,bbox_color=clr)
    # img1 = bbv.add_multiple_labels(img1,labels,bboxes,top=False,text_bg_color=clr)
    return img1

def read_class_map(path):
    cl_map = pd.read_csv(path,index_col=0).to_dict()['label']
    return cl_map

def clear_temp_dirs():
    temp_saved = glob.glob('saved_temp/*')
    for f in temp_saved:
        os.remove(f)
    temp_vids = glob.glob('video_preds/temp/*')
    for f in temp_vids:
        os.remove(f)
    

def convert_avi_to_mp4(input_avi, output_mp4):
    video = VideoFileClip(input_avi)
    video.write_videofile(output_mp4)
    os.remove(input_avi)


def get_msg_data(message):
    """prints and saves message information
    format: user_name|massage text/type of message|%d/%m/%Y %H:%M:%S
    """
    msg_types = {'photo':'PHOTO','video':'VIDEO','audio':'audio','sticker':'sticker','video_note':'kruzhok','voice':'voice'}
    if os.path.isfile('msg_logs.csv'):
        df = pd.read_csv('msg_logs.csv', index_col=0)
    else:
        df = pd.DataFrame(columns = ['user_id', 'user_name', 'message_info', 'date'])
    now = datetime.now(pytz.timezone('Europe/Moscow'))
    now = now.strftime("%d/%m/%Y %H:%M:%S")
    user_name = message.from_user.username
    user_id = message.from_user.id
    if message.content_type == 'text':
        msg_info = message.text
    elif message.content_type in msg_types:
        msg_info = msg_types[message.content_type]
    else:
        msg_info = 'something went wrong'

    df.loc[len(df.index)] = [user_id, user_name, msg_info, now]
    df.to_csv('msg_logs.csv')
    print('got new message: {} | {} | {}'.format(user_name,msg_info,now))