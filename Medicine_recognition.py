from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import cv2

import threading

from PIL import ImageFont, ImageDraw, Image
from tflite_runtime.interpreter import Interpreter

from gtts import gTTS
import googletrans
from playsound import playsound
import os

#include <opencv2/freetype.hpp>


language = 'zh'
translator = googletrans.Translator()

DELAY_TIME = 3.0

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def voice_play(sound):
    playsound(sound)

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # print('input_tensor=',input_tensor)
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]



def main():
  global labels
  global label_id
  
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of model.', required=True)
  parser.add_argument(
      '--video', help='Video number', required=False, type=int, default=0)

  args = parser.parse_args()

  model_file = args.model + '/model.tflite'

  labels_file = args.model + '/labels.txt'

  labels = load_labels(labels_file)

  jud=input('Input your country (with your mother tongue ):')
  text_language = translator.detect(jud)

  if(os.path.isfile('label_'+str(0)+'_'+str(text_language.lang)+'.mp3') != True):
    for i in range(len(labels)):
        text_list = labels[i].split(' ')
        text_2 = translator.translate(text_list[1], dest=str(text_language.lang)).text    
        speech = gTTS(text = text_2, lang = text_language.lang, slow = False)
        speech_file = 'label_'+str(i)+'_'+str(text_language.lang)+'.mp3'
        speech.save(speech_file)
  text_list_2=[]
  for i in range(len(labels)):
        text_list = labels[i].split(' ')
        text_2_2 = translator.translate(text_list[1], dest=str(text_language.lang)).text
        text_list_2.append(text_2_2)
  
  interpreter = Interpreter(model_file)
  interpreter.allocate_tensors()

  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
    
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  cap = cv2.VideoCapture(args.video)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  key_detect = 0
  times=1
  t_flag = 0
  PRE_TIME = time.time()
  
  while (key_detect==0):
    ret,image_src =cap.read()

    frame_width=image_src.shape[1]
    frame_height=image_src.shape[0]

    cut_d=int((frame_width-frame_height)/2)
    crop_img=image_src[0:frame_height,cut_d:(cut_d+frame_height)]

    image=cv2.resize(crop_img,(224,224),interpolation=cv2.INTER_AREA)

    start_time = time.time()
    if (times==1):
      results = classify_image(interpreter, image)
      elapsed_ms = (time.time() - start_time) * 1000
      label_id, prob = results[0]

    if(labels[label_id] == '3 Other'):
        img = np.zeros((200, 600, 3), np.uint8)
        text = text_list_2[label_id]
        img[:] = (87, 207, 227)
        fontPath = "C:\Windows\Fonts\kaiu.ttf"
        font = ImageFont.truetype(fontPath, 100)
        imgPil = Image.fromarray(img)
        draw = ImageDraw.Draw(imgPil)
        draw.text((30, 30),  text, font = font, fill = (0, 0, 0))
        img = np.array(imgPil)
        cv2.imshow('My Image', img)

    else:
      if((t_flag == 0) and (time.time() - PRE_TIME > DELAY_TIME)):
        PRE_TIME = time.time()
        sound_file = 'label_'+ str(label_id)+'_'+str(text_language.lang)+'.mp3'
        t_voice_play = threading.Thread(target = voice_play, args=(sound_file,))
        t_voice_play.start()
        t_flag = 1
      img = np.zeros((200, 600, 3), np.uint8)
      text = text_list_2[label_id]
      img[:] = (87, 207, 227)
      fontPath = "C:\Windows\Fonts\kaiu.ttf"
      font = ImageFont.truetype(fontPath, 100)
      imgPil = Image.fromarray(img)
      draw = ImageDraw.Draw(imgPil)
      draw.text((30, 30),  text, font = font, fill = (0, 0, 0))
      img = np.array(imgPil)
      cv2.imshow('My Image', img)


    if (t_flag == 1):
      t_voice_play.join()
      t_flag = 0

    times=times+1
    if (times>10):
      times=1

    cv2.imshow('Detecting....',crop_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      key_detect = 1

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
