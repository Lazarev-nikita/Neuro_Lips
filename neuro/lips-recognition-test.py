import sys
import math
import numpy as np
import cv2
import dlib
import tensorflow as tf

import os.path
import json

class Settings:
    path: str
    config: None

    def __init__(self, path):
        self.config = {}
        self.path = path
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.config = json.load(f)

    def get(self):
        return self.config

    def set(self, config):
        if (config is None):
            config = {}

        self.config = config
        self.save()

    def has_value(self, name):
        return name in self.config

    def get_value(self, name: str, def_value: any = None):
        return self.get_value_cfg(self.config, name, def_value)

    def get_value_cfg(self, cgf, name: str, def_value: any = None):
        if name not in cgf:
            cgf[name] = def_value
        return cgf[name]

    def set_value(self, name: str, value: any):
        self.config[name] = value
        self.save()

    def save(self):
        with open(self.path, 'w', encoding="utf-8") as fo:
            fo.write(json.dumps(self.config, indent=2, ensure_ascii=False))

FONT = cv2.FONT_HERSHEY_COMPLEX
TRAINED_MODEL_PATH = './models-final'
APP_SETTINGS_PATH = './lips-reco-settings.json'

WIDTH = 128
HEIGHT = 128

LIPS_MUTE_MAX_DELTA = 2
MUTE_SYMBOL = ' '


class Point:
    x: int
    y: int

    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def distance(self, p):
        dx = self.x - p.x
        dy = self.y - p.y
        return math.sqrt(dx**2 + dy**2)


class Rect:
    tl: Point
    br: Point

    def __init__(self, tl: Point, br: Point):
        self.tl = tl
        self.br = br

    def center(self) -> Point:
        return Point((self.tl.x + self.br.x) / 2, (self.tl.y + self.br.y) / 2)

    def cx(self):
        return self.br.x - self.tl.x

    def cy(self):
        return self.br.y - self.tl.y


class FramesCache:
    FRAMES: int = 12
    CACHE_RESIZE = 8192
    start: int = -1
    end: int = -1
    framesCache: list = None

    def __init__(self):
        self.framesCache = [None] * self.CACHE_RESIZE

    def add_frame(self, frame):
        if self.start < 0:
            self.end = 0
        self.framesCache[self.end] = frame
        self.end += 1
        self.start = max(0, self.end - self.FRAMES)
        if self.end >= len(self.framesCache):
            self.framesCache = self.framesCache[self.start:]
            self.framesCache.extend([None] * self.CACHE_RESIZE)
            self.start = 0
            self.end = self.start +  self.FRAMES

    def is_ready(self):
        return self.end - self.start == self.FRAMES

    def get_frames(self):
        return self.framesCache[self.start : self.end], self.start, self.end


appSettings = Settings(APP_SETTINGS_PATH)
SYMBOLS: dict = appSettings.get_value('symbols', dict())
CAMERA_INDEX: int = appSettings.get_value('camera', 0)


# prepare face points detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\\Projects\\git\\lip-reading-deeplearning\\data\\shape_predictor_68_face_landmarks.dat")

points = list()
points.extend(range(2, 15))     # bottom part of face contour
points.extend(range(30, 36))    # bottom part of nose
points.extend(range(48, 68))    # bottom part of nose

# load neuro model
model = tf.keras.models.load_model(TRAINED_MODEL_PATH)

# connect to camera
# camera = cv2.VideoCapture(CAMERA_INDEX)

# src_path = './speach-text/speach-text.mp4'
# src_path = './speach-2/speach.mp4'
src_path = './speach-1/two.mp4'
srcVideo = cv2.VideoCapture(src_path)
srcFrameCount = int(srcVideo.get(cv2.CAP_PROP_FRAME_COUNT))
frameW = int(srcVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
frameH = int(srcVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f'Analyze file: {srcVideo}')
print(f'Total frames to analyze: {srcFrameCount}')

dst_path = './speach-1/analyse-result-speach-text.avi'
dst_text = './speach-1/analyse-result-speach-text.txt'
out_stream = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), 30, (frameW, frameH))
out_text = open(dst_text, 'w', encoding="utf-8")

print(f'Save result to: {dst_path}, {dst_text}')

cropRect = Rect(Point(-1, -1), Point(-1, -1))

framesCache = FramesCache()
currFrame = 0
oldProgress = 0
lastSymbol = None
lastStart = -1
while True:
    ret, frame = srcVideo.read()
    if not ret:
        break

    imgShow = frame.copy()

    h, w, c = frame.shape
    grayFrame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    faces = detector(grayFrame)
    if len(faces) > 0: #face detected
        face = faces[0]
        # cv2.rectangle(imgShow, (face.left(), face.top()), (face.right(), face.bottom()), (0, 200, 255), 1)

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
            if n in [62, 66]:
                cv2.circle(imgShow, (p.x, p.y), 1, (0, 255, 0), 1)
                # cv2.circle(imgShow, (p.x, p.y), 1, (0, 0, 255), 1)
            else:
                cv2.circle(imgShow, (p.x, p.y), 1, (0, 255, 0), 1)

        m62 = landmarks.part(62)
        m66 = landmarks.part(66)
        p62 = Point(m62.x, m62.y)
        p66 = Point(m66.x, m66.y)

        lipsDelta = p62.distance(p66)

        if lipsDelta < LIPS_MUTE_MAX_DELTA:
            cv2.putText(imgShow, 'Mute', (w//2, h - 50), FONT, 0.6, (0, 255, 0), 1)
            if lastSymbol != MUTE_SYMBOL:
                lastSymbol = MUTE_SYMBOL
                out_text.write(lastSymbol)
                out_text.flush()
                print(f'Detected: Mute ({currFrame+1} of {srcFrameCount})')

        frameRect = Rect(Point(int(np.amin(marks, axis=1)[0]),
                               int(np.amin(marks, axis=1)[1])),
                         Point(int(np.amax(marks, axis=1)[0]),
                               int(np.amax(marks, axis=1)[1])))

        frameCenter: Point = frameRect.center()
        sz = max(frameRect.cx(), frameRect.cy())

        cropRect.tl.x = int(frameCenter.x - sz // 2)
        if cropRect.tl.x < 0:
            cropRect.tl.x = 0
        cropRect.br.x = cropRect.tl.x + sz
        if cropRect.br.x >= w:
            cropRect.br.x = w-1
            cropRect.tl.x = cropRect.br.x - sz

        cropRect.tl.y = int(frameCenter.y - sz // 2)
        if cropRect.tl.y < 0:
            cropRect.tl.y = 0
        cropRect.br.y = cropRect.tl.y + sz
        if cropRect.br.y >= h:
            cropRect.br.y = h-1
            cropRect.tl.y = cropRect.br.y - sz
    else:
        print(f'Frame {currFrame}: face not detected')

    if cropRect.tl.x >= 0 and cropRect.br.x >= 0 and cropRect.tl.y >= 0 and cropRect.br.y >= 0:
        mouthFrame = frame[cropRect.tl.y:cropRect.br.y, cropRect.tl.x:cropRect.br.x]
        resizedFrame = cv2.resize(mouthFrame, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
        resizedFrame = resizedFrame.astype(np.float32)
        resizedFrame /= 255.
        framesCache.add_frame(resizedFrame)
        if lipsDelta >= LIPS_MUTE_MAX_DELTA and framesCache.is_ready():
            framesList, frStart, frEnd = framesCache.get_frames()
            preparedFrames = np.array(framesList)[..., [2, 1, 0]]
            images = []
            images.append(preparedFrames)
            np_img = np.array(images)
            preds = model(np_img, training = False)
            preds_arr = np.array(preds)
            i = np.argmax(preds_arr)
            label = SYMBOLS[str(i)]
            cv2.putText(imgShow, label, (w//2, h - 50), FONT, 0.6, (0, 255, 0), 1)

            if lastSymbol != label:
                if lastSymbol is not None and lastSymbol != MUTE_SYMBOL:
                    out_text.write('-')
                    out_text.flush()
                lastSymbol = label
                lastStart = frStart
                out_text.write(lastSymbol)
                out_text.flush()

            elif lastStart + framesCache.FRAMES <= frStart:
                if lastSymbol is not None and lastSymbol != MUTE_SYMBOL:
                    out_text.write('-')
                    out_text.flush()
                lastSymbol = label
                lastStart = frStart
                out_text.write(lastSymbol)
                out_text.flush()

            print(f'Detected: {label} ({currFrame+1} of {srcFrameCount})')


    # cv2.imshow(winname="Face", mat=imgShow)
    out_stream.write(imgShow)

    progress = int (((currFrame+1) * 100) / srcFrameCount)
    if progress != oldProgress:
        print(f'Analyze progress: {progress}% ({currFrame+1} of {srcFrameCount})')
        oldProgress = progress
    currFrame += 1

    # Exit when escape is pressed
    # key = cv2.waitKey(delay=1)
    # if key == 27:
    #     break

print('Analyze completed')

out_stream.release()
out_text.close()
srcVideo.release()


print('Application stopped')

# Close all windows
# cv2.destroyAllWindows()