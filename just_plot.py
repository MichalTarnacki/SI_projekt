import noisereduce
import pyaudio
import numpy as np
import time
import wave
import matplotlib.pyplot as plt
from joblib import load
from scipy.io.wavfile import write
import src.data_engineering.spectrogram as sp

import macros

# open stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

CHUNK = 320 # RATE / number of updates per second

RECORD_SECONDS = 20


# use a Blackman window
window = np.blackman(CHUNK)

x = 0

state = 'in'
prev_state = 'in'
def soundPlot(stream, ax1, ax2, model, scaler):
    global state, prev_state
    t1=time.time()
    waveData = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

    waveData_t = waveData.copy()

    #waveData =  [float(i)/32767.0 * 1000 for i in waveData]


    npArrayData_t = np.array(waveData_t)
    indata_t = npArrayData_t*window
    fftData_t = noisereduce.reduce_noise(indata_t, 44100)
    fftData_t = np.abs(np.fft.rfft(fftData_t))
    fftData_t = fftData_t[fftData_t.__len__() - 161:]


    #waveData = wave.#.unpack("%dh"%(CHUNK), data)
    npArrayData = np.array(waveData)
    indata = npArrayData*window
    #write('test.wav', RATE, indata)
    fftData = noisereduce.reduce_noise(indata, 44100)
    fftData = np.abs(np.fft.rfft(fftData))
    fftData = fftData[fftData.__len__() - 160:]
    fftTime = np.fft.rfftfreq(CHUNK, 1. / RATE)
    fftTime = fftTime[fftTime.__len__() - 161:]

    #if fftData.__len__() >= 160:
    fftData = np.append(fftData, [1 if prev_state == 'in' else -1])
    state = model.predict(scaler.transform(fftData.reshape(-1, 1).T))
    which = fftData[1:].argmax() + 1

    #Plot time domain
    ax1.cla()
    ax1.plot(indata, 'g' if prev_state == 'in' else 'r')
    ax1.grid()
    if np.mean(fftData) > 0:
        ax1.set_title('in' if prev_state == 'in' else 'out')
    else:
        ax1.set_title('none')
    ax1.axis([0,CHUNK,-5000,5000])

    #Plot frequency domain graph
    ax2.cla()
    ax2.plot(fftTime,fftData_t, 'g' if prev_state == 'in' else 'r')
    ax2.grid()
    ax2.axis([0,5000,0,10**6])
    plt.pause(0.01)
    print("took %.02f ms"%((time.time()-t1)*1000))
    # use quadratic interpolation around the max
    if which != len(fftData)-1:
        y0,y1,y2 = np.log(fftData[which-1:which+2:])
        x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
        # find the frequency and output it
        thefreq = (which+x1)*RATE/CHUNK
        print ("The freq is %f Hz." % (thefreq))
    else:
        thefreq = which*RATE/CHUNK
        print( "The freq is %f Hz." % (thefreq))

    prev_state = state

def new_realtime(modelfile):
    model = load(f'{macros.model_path}{modelfile}.joblib')
    scaler = load(f'{macros.model_path}{modelfile}_scaler.joblib')
    p=pyaudio.PyAudio()
    stream=p.open(format=FORMAT,channels=1,rate=RATE,input=True,
                  frames_per_buffer=CHUNK)

    plt.ion()
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for i in range(0, RATE // CHUNK * RECORD_SECONDS):
        soundPlot(stream, ax1, ax2, model, scaler)


    stream.stop_stream()
    stream.close()
    p.terminate()