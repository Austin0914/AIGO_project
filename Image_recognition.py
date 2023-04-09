from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import cv2

import threading

from PIL import Image
from tflite_runtime.interpreter import Interpreter

from gtts import gTTS
import googletrans
from playsound import playsound
import os


translator = googletrans.Translator()

Condition_1 = '0 Correct'
Condition_2 = '1 Wrong'
Condition_3 = '2 Wrong'
Condition_4 = '3 Wrong'

DELAY_TIME = 3.0

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


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

def voice_play(sound):
  playsound(sound)

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
  speech_file = 'wrong_'+str(text_language.lang)+'.mp3'
  if(os.path.isfile(speech_file) != True):
    text='hold wrong medicine box'
    text_2 = translator.translate(text, dest=str(text_language.lang)).text    
    speech = gTTS(text = text_2, lang = text_language.lang, slow = False)
    speech.save(speech_file)
  
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

    if(labels[label_id]==Condition_1):
        cv2.rectangle(crop_img, (5, 5), (475, 475), (0, 255, 0), 5)
    else:
        if((time.time() - PRE_TIME > DELAY_TIME) and (t_flag == 0)):
            PRE_TIME = time.time()
            sound_file='wrong_'+str(text_language.lang)+'.mp3'
            t_voice_play = threading.Thread(target = voice_play, args=(sound_file,))
            t_voice_play.start()
            t_flag = 1
        cv2.rectangle(crop_img, (5, 5), (475, 475), (0, 0, 255), 10)
        cv2.line(crop_img,(5,5),(475,475),(0,0,255),10)
        cv2.line(crop_img,(475,5),(5,475),(0,0,255),10)

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