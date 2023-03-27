import pydub
import threading
import os
import pathlib
import datetime
import cv2
import numpy as np
import dlib
import pyaudio
import wave
import time
import sys
import collections
import math
from modules import settings
from modules import camera

import os.path
import json

def get_segments(audio_path: str):
    wav = pydub.AudioSegment.from_wav(audio_path)
    segments = pydub.silence.detect_nonsilent(wav, min_silence_len=400, silence_thresh = wav.dBFS-10)
    print(f'segments found = {len(segments)}; {segments}')

    #get average duration:
    avg_count = len(segments)
    avg_sum = 0
    start_segment = 0
    out_segments = list()
    for s in segments:
        if s[0] == 0:
            start_segment += 1
            avg_count -= 1
            print(f'skip noise segment at begin: {s}')
            continue
        out_segments.append((s[0], s[1], s[1] - s[0]))
        avg_sum += (s[1] - s[0])
    avg_dur = int(math.ceil(avg_sum / avg_count))

    return {'avg': avg_dur, 'segments': out_segments}


def get_sym_ext(filename: str) -> (str, str, str):
    sym, p = filename.split('_')
    idx, ext = p.split('.')
    return sym, idx, ext


# RECORDS_PATH = './result-data-n-1'
RECORDS_PATH = './result-data-n'
# RECORDS_PATH = './result-data-pl-1'
# RECORDS_PATH = './result-data-pl-2'
# SPLIT_RECORDS_OUT_PATH = './split-result-n-1'
SPLIT_RECORDS_OUT_PATH = './split-result-n'
# SPLIT_RECORDS_OUT_PATH = './split-result-pl-1'
# SPLIT_RECORDS_OUT_PATH = './split-result-pl-2'

FRAMES_CROP_COUNT = 12

pathlib.Path(SPLIT_RECORDS_OUT_PATH).mkdir(parents=True, exist_ok=True)

generatorSettings = settings.Settings('record-settings.json')
SYMBOLS = generatorSettings.get_value('symbols')

# prepare face points detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Projects\\my\\python\\lips_data\\shape-predictor\\shape_predictor_68_face_landmarks.dat")

points = list()
points.extend(range(2, 15))     # bottom part of face contour
points.extend(range(30, 36))    # bottom part of nose
points.extend(range(48, 68))    # bottom part of nose

WIDTH = 128
HEIGHT = 128

files_list = os.listdir(RECORDS_PATH)

# source_data = {
#     'A': {
#         '1': {
#             'wav': 'A_1.wav',
#             'avi': 'A_1.avi',
#             'avg': average_duration_value,
#             'segments': [found audio segments],
#         }
#     }
# }
total_files_count = 0
source_data = collections.defaultdict(dict)
print('Collect input files...')
for f in files_list:
    sym, idx, ext = get_sym_ext(f)
    sym_data = source_data[sym]
    if idx not in sym_data:
        sym_data[idx] = collections.defaultdict(dict)
    sym_data[idx][ext] = f
    total_files_count += 1

print('Collect audio segments for input files')
num = 1
sum_avg = 0
count_avg = 0
audio_count = int(total_files_count/2)
for sym in source_data:
    print(f'search segments for {sym}')
    sym_data = source_data[sym]
    for idx in sym_data:
        idx_data = sym_data[idx]
        src_audio = os.path.join(RECORDS_PATH, idx_data['wav'])
        print(f'Checking file {num} of {audio_count}: {src_audio}')

        audio_segments = get_segments(src_audio)
        print(f'Average segment duration: {audio_segments["avg"]}')
        idx_data['avg'] = audio_segments['avg']
        idx_data['segments'] = audio_segments['segments']
        sum_avg += audio_segments['avg']
        count_avg += 1
        num += 1

avg = sum_avg / count_avg
print(f'Average segment duration over all files: {avg}')

X_left_crop = -1
X_right_crop = -1
Y_left_crop = -1
Y_right_crop = -1

num = 1
video_count = int(total_files_count/2)
for sym in source_data:
    print(f'split video files for {sym}...')
    sym_data = source_data[sym]
    for idx in sym_data:
        idx_data = sym_data[idx]
        src_video = os.path.join(RECORDS_PATH, idx_data['avi'])
        audio_segments = idx_data['segments']
        print(f'Split video file {src_video}...')
        src_cap = cv2.VideoCapture(src_video)

        for segment_idx, s in enumerate(audio_segments):  # (start, end, duration)
            print(f'segment extract ({s[0]}, {s[1]}) ...')
            s_start = int(s[0] + s[2] / 2 - avg / 2)
            s_end = s_start + avg

            dst_video = os.path.join(SPLIT_RECORDS_OUT_PATH, '{symbol}_{idx}_{iter}.avi'.format(symbol=sym, idx=str(idx).zfill(2), iter=str(segment_idx).zfill(2)))
            print(f'Record video file {dst_video}')
            crop_stream = cv2.VideoWriter(dst_video, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), 30, (WIDTH, HEIGHT))

            # seach for frames diapason:
            src_cap.set(cv2.CAP_PROP_POS_MSEC, int(s_start + avg / 2))
            frmIndexCenter = int(src_cap.get(cv2.CAP_PROP_POS_FRAMES))
            frmIndexStart = frmIndexCenter - int(FRAMES_CROP_COUNT/2)
            if frmIndexStart < 0:
                frmIndexStart = 0

            print(f'Start crop from frame index = {frmIndexStart}')
            src_cap.set(cv2.CAP_PROP_POS_FRAMES, frmIndexStart)
            croppedFrames = 0
            while croppedFrames < FRAMES_CROP_COUNT:
                ret, frame = src_cap.read()
                if not ret:
                    break
                croppedFrames += 1

                h, w, c = frame.shape
                grayFrame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
                faces = detector(grayFrame)
                if len(faces) > 0: #face detected
                    face = faces[0]
                    # Create landmark object
                    landmarks = predictor(image=grayFrame, box=face)
                    marks = np.zeros((2, len(points)))
                    # Loop through all the points
                    co = 0
                    # Specific for the mouth.
                    for n in points:
                        p = landmarks.part(n)
                        A = (p.x, p.y)
                        marks[0, co] = p.x
                        marks[1, co] = p.y
                        co += 1

                    frameRect = (int(np.amin(marks, axis=1)[0]),
                                 int(np.amin(marks, axis=1)[1]),
                                 int(np.amax(marks, axis=1)[0]),
                                 int(np.amax(marks, axis=1)[1]))
                    frameCenter = ((frameRect[0] + frameRect[2]) // 2, (frameRect[1] + frameRect[3]) // 2)
                    size = (frameRect[2] - frameRect[0], frameRect[3] - frameRect[1])
                    sz = max(size)
                    X_left_crop = int(frameCenter[0] - sz // 2)
                    if X_left_crop < 0:
                        X_left_crop = 0
                    X_right_crop = X_left_crop + sz
                    if X_right_crop >= w:
                        X_right_crop = w-1
                        X_left_crop = X_right_crop - sz

                    Y_left_crop = int(frameCenter[1] - sz // 2)
                    if Y_left_crop < 0:
                        Y_left_crop = 0
                    Y_right_crop = Y_left_crop + sz
                    if Y_right_crop >= h:
                        Y_right_crop = h-1
                        Y_left_crop = Y_right_crop - sz
                else:
                    print(f'Face not found, use previous rect: ({X_left_crop}, {Y_left_crop}, {X_right_crop}, {Y_right_crop})')

                if X_left_crop >= 0 and X_right_crop >= 0 and Y_left_crop >= 0 and Y_right_crop >= 0:
                    mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop]
                    resized = cv2.resize(mouth, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
                    crop_stream.write(resized)

            print(f'{dst_video} frames: {croppedFrames}')
            crop_stream.release()

        num += 1

print('Video split done')