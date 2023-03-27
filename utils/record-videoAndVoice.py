import os
import os.path
import json
import pathlib
import datetime
import cv2
import pyaudio
import wave

class Settings:
    path: str = None
    config: dict = None

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

    def get_value_cfg(self, cfg: dict, name: str, def_value: any = None):
        if name not in cfg:
            cfg[name] = def_value
        return cfg[name]

    def set_value(self, name: str, value: any):
        self.config[name] = value
        self.save()

    def save(self):
        with open(self.path, 'w', encoding="utf-8") as fo:
            fo.write(json.dumps(self.config, indent=2, ensure_ascii=False))


def select_audio(p: pyaudio.PyAudio()):
    print('Select microphone from list:')
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d['maxInputChannels'] > 0:
            print(i, d['name'])
    return int(input())


def wave_callback (in_data, frame_count, time_info, status):
    if wave_stream is not None:
        wave_stream.writeframes (in_data)
    return (in_data, pyaudio.paContinue)


def start_record(currSymbol: str, pa: pyaudio.PyAudio()):
    video_path = os.path.join(RECORDS_OUT_PATH, VIDEO_MASK.format(symbol=currSymbol))
    vs = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), 30, (frameWidth, frameHeight))

    audio_path = os.path.join(RECORDS_OUT_PATH, WAV_MASK.format(symbol=currSymbol))
    ws = wave.open (audio_path, "wb")
    ws.setnchannels (AUDIO_CHANNELS)
    ws.setsampwidth (pa.get_sample_size(AUDIO_FORMAT))
    ws.setframerate (AUDIO_RATE)

    ms = pa.open(format = pa.get_format_from_width(ws.getsampwidth()),
                               channels = ws.getnchannels(),
                               rate = ws.getframerate(),
                               input=True,
                               stream_callback=wave_callback)
    ms.start_stream()

    return vs, ws, ms


def stop_record(vs, ws, ms):
    if ms is not None:
        ms.stop_stream()
        ms.close()
        ms = None
    if ws is not None:
        ws.close()
        ws = None
    if vs is not None:
        vs.release()
        vs = None
    return vs, ws, ms


MICROPHONE_INDEX = -1
CAMERA_INDEX = -1
AUDIO_CHUNK = 1024
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 2
AUDIO_RATE = 44100

class GeneratorState:
    RECORD_NOT_STARTED  = 1
    RECORD_IN_PROGRESS_1  = 2
    RECORD_IN_PROGRESS_2  = 3
    RECORD_SAVING       = 4
    DONE            = 5


RECORDS_OUT_PATH = './result-data'
pathlib.Path(RECORDS_OUT_PATH).mkdir(parents=True, exist_ok=True)

generatorSettings = Settings('record-settings.json')
MICROPHONE_INDEX = generatorSettings.get_value('last-microphone', -1)
CAMERA_INDEX = generatorSettings.get_value('last-camera', -1)
FONT = cv2.FONT_HERSHEY_COMPLEX
SYMBOLS: int = generatorSettings.get_value('symbols', list())
REPEATS: int = generatorSettings.get_value('repeats', 20)
currSymbolIndex: int = generatorSettings.get_value('last-symbol-index', 0)
iteration: int = generatorSettings.get_value('last-iteration', 0)
generatedDataMap: dict = generatorSettings.get_value('generated-data', {})

if iteration >= REPEATS:
    iteration = 0
    currSymbolIndex += 1
    generatorSettings.set_value('last-symbol-index', currSymbolIndex)
    generatorSettings.set_value('last-iteration', iteration)

if currSymbolIndex >= len(SYMBOLS):
    print('All symbols readed. Exit')
    exit()

pAudio = pyaudio.PyAudio()

if MICROPHONE_INDEX == -1:
    MICROPHONE_INDEX = select_audio(pAudio)
    generatorSettings.set_value('last-microphone', MICROPHONE_INDEX)

if CAMERA_INDEX == -1:
    CAMERA_INDEX = 0 #camera.get_camera()
    generatorSettings.set_value('last-camera', CAMERA_INDEX)

print(f'Microphone: {MICROPHONE_INDEX}')
print(f'Camera: {CAMERA_INDEX}')

# connect to camera
camera = cv2.VideoCapture(CAMERA_INDEX)
frameWidth  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
frameHeight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

video_stream = None
wave_stream = None
microphone_stream = None

WAV_MASK = '{symbol}.wav'
VIDEO_MASK = '{symbol}.avi'




font = cv2.FONT_HERSHEY_COMPLEX

genStatus = GeneratorState.RECORD_NOT_STARTED
currSymbol = SYMBOLS[currSymbolIndex]
start_timestamp = None
while True:
    ret, frame = camera.read()

    if genStatus == GeneratorState.RECORD_NOT_STARTED:
        cv2.putText(frame, 'Press enter to start recording', (50, frameHeight - 70), font, 0.8, (0, 0, 255), 1)
        cv2.putText(frame, 'Next symbol to read: ' + currSymbol, (50, frameHeight - 40), font, 0.7, (0, 255, 255), 1)
    elif genStatus == GeneratorState.RECORD_SAVING:
        cv2.putText(frame, '...', (frameWidth//2, frameHeight - 50), font, 1, (0, 0, 255), 2)
        video_stream, wave_stream, microphone_stream = stop_record(video_stream, wave_stream, microphone_stream)
        if currSymbolIndex >= len(SYMBOLS):
            genStatus = GeneratorState.DONE
            break
        else:
            currSymbol = SYMBOLS[currSymbolIndex]
            genStatus = GeneratorState.RECORD_NOT_STARTED
            # video_stream, wave_stream, microphone_stream = start_record(currSymbol, pAudio)
            # genStatus = GeneratorState.RECORD_IN_PROGRESS_1
    else:
        if video_stream is not None:
            video_stream.write(frame)

        curr_time = datetime.datetime.now()
        if genStatus == GeneratorState.RECORD_IN_PROGRESS_1:
            cv2.putText(frame, currSymbol, (frameWidth//2, frameHeight - 50), font, 0.6, (0, 255, 0), 1)
            if start_timestamp is None:
                start_timestamp = datetime.datetime.now()
            else:
                delta = curr_time - start_timestamp
                if delta.total_seconds() >= 0.5:
                    genStatus = GeneratorState.RECORD_IN_PROGRESS_2
                    start_timestamp = None
        elif genStatus == GeneratorState.RECORD_IN_PROGRESS_2:
            cv2.putText(frame, currSymbol, (frameWidth//2, frameHeight - 50), font, 1, (0, 0, 255), 2)
            if start_timestamp is None:
                start_timestamp = datetime.datetime.now()
            else:
                delta = curr_time - start_timestamp
                if delta.total_seconds() >= 0.5:
                    start_timestamp = None
                    iteration += 1
                    if iteration >= REPEATS:
                        iteration = 0
                        currSymbolIndex += 1
                        genStatus = GeneratorState.RECORD_SAVING
                        generatorSettings.set_value('last-symbol-index', currSymbolIndex)
                    else:
                        genStatus = GeneratorState.RECORD_IN_PROGRESS_1
                    generatorSettings.set_value('last-iteration', iteration)

    # show the image
    cv2.putText(frame, 'Press ESC to exit', (50, frameHeight - 10), font, 0.6, (0, 255, 255), 1)
    cv2.imshow(winname="Face", mat=frame)

    # Exit when escape is pressed
    key = cv2.waitKey(delay=1)
    if key == 27:
        video_stream, wave_stream, microphone_stream = stop_record(video_stream, wave_stream, microphone_stream)
        break
    elif key == 13:
        if genStatus == GeneratorState.RECORD_NOT_STARTED:
            video_stream, wave_stream, microphone_stream = start_record(currSymbol, pAudio)
            genStatus = GeneratorState.RECORD_IN_PROGRESS_1

if pAudio is not None:
    pAudio.terminate()

if camera is not None:
    camera.release()

print ("* recording done!")

# Close all windows
cv2.destroyAllWindows()