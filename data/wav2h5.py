from scipy.io import wavfile
import tables
import numpy
import sys

#read data from wav
fs, data = wavfile.read("cl4ac/data/wav_files/test.wav")

#save_to acoular h5 format
acoularh5 = tables.open_file("cl4ac/data/logspectrogram/test.wav", mode = "w", title = "test")
acoularh5.create_earray('/','time_data', atom=None, title='', filters=None, \
                         expectedrows=100000, \
                         byteorder=None, createparents=False, obj=data)
acoularh5.set_node_attr('/time_data','sample_freq', fs)
acoularh5.close()