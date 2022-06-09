from collections import namedtuple

# Faiss
DIMENSION = 256#192

DB_ROOT = '/mnt/c/Users/phudh/Desktop/src/dialog_system/Identify_speaker/speaker_id'

# Inference
THRESHOLD = 0
SCORE = 0
SPEAKER = 1

UN_IDENTIFIED = 'UN_IDENTIFIED'
IDENTIFIED = 'IDENTIFIED'


# Consumers
AudioFormat = namedtuple('AudioFormat', 'rate channels width')
DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WIDTH = 2
DEFAULT_FORMAT = AudioFormat(DEFAULT_RATE, DEFAULT_CHANNELS, DEFAULT_WIDTH)