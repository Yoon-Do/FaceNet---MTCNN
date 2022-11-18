from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from email.mime import image
from secrets import choice
from pathlib import Path

import tensorflow.compat.v1 as tf
from imutils.video import VideoStream


import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import time
import random


def main():
    tf.disable_v2_behavior()
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()
    prev_frame_time = 0
    new_frame_time = 0
   

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            people_detected = set()
            person_detected = collections.Counter()

            done = False

            cap  = VideoStream(src=0).start()
            print('Enter your choice :')
            choice = input()
            if choice == '1':
                print('Enter the name of the person :')
                name_learning = input()
                person_detected.__add__(name_learning)
                if not os.path.exists('Dataset/FaceData/processed/'+str(name_learning)):
                    os.makedirs('Dataset/FaceData/processed/'+str(name_learning))
            while (True):
                frame = cap.read()
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)


                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            print(bb[i][3]-bb[i][1])
                            print(frame.shape[0])
                            print((bb[i][3]-bb[i][1])/frame.shape[0])
                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))


                                if best_class_probabilities > 0.6:
                                    
                                    cv2.rectangle(frame, (bb[i][0] - 30, bb[i][1]- 60), (bb[i][2] + 30, bb[i][3] + 20), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    crop_image = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]]
                                    crop_image = cv2.resize(crop_image, (160, 160))
                                    if person_detected[best_name] > 30:
                                        cv2.imwrite("Dataset/FaceData/processed/"+str(best_name)+"/" +str(random.randint(1,1000000000))  + ".png", crop_image)
                                        person_detected[best_name] = 0
                                        done = True
                                        
                                        
                                    name = class_names[best_class_indices[0]]
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame,'FPS: ' + fps  , (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 0, 255), thickness=1, lineType=2)
                                    person_detected[best_name] += 1
                                    
                                else:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    if choice == '1':
                                        crop_image = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]]
                                        crop_image = cv2.resize(crop_image, (160, 160))
                                     #   if person_detected[name_learning] > 1:
                                     #        cv2.imwrite("Dataset/FaceData/processed/"+str(name_learning)+"/" +str(random.randint(1,1000000000))  + ".png", crop_image)
                                     #        person_detected[name_learning] = 0 */
                                        cv2.putText(frame, name_learning + "_Learning", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (255, 255, 255), thickness=1, lineType=2)
                                        cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (255, 255, 255), thickness=1, lineType=2)
                                        cv2.putText(frame,'FPS: ' + fps  , (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 0, 255), thickness=1, lineType=2)
                                        person_detected[name_learning] += 1
                                    else:
                                        cv2.putText(frame, "Unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (255, 255, 255), thickness=1, lineType=2)
                                        cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (255, 255, 255), thickness=1, lineType=2)
                                        cv2.putText(frame,'FPS: ' + fps  , (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 0, 255), thickness=1, lineType=2)

                except:
                    pass
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Chào mừng " + str(best_name))
                    #os.system('python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 500')
                    break
            cap.release()
            cv2.destroyAllWindows()


main()