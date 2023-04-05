import yaml
import voicerss_tts
import wave

with open('configuration.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

voice = voicerss_tts.speech({
    'key': config['api_key'],
    'hl': config['language'],
    'v': config['voice'],
    'src': 'Hello, world!',
    'r': config['speed'],
    'c': config['codec'],
    'f': config['format'],
    'ssml': 'false',
    'b64': 'false'
})

print('starting')

with wave.open('output.wav', 'wb') as wav_file:
    wav_file.setnchannels(1)  # mono
    wav_file.setsampwidth(2)  # 2 bytes per sample
    wav_file.setframerate(44100)
    wav_file.writeframes(voice['response'])

print('done')