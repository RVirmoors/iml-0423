import yaml
import voicerss_tts

with open('configuration.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

voice = voicerss_tts.speech({
    'key': config['api_key'],
    'hl': 'en-us',
    'v': 'Linda',
    'src': 'Hello, world!',
    'r': '0',
    'c': 'mp3',
    'f': '44khz_16bit_stereo',
    'ssml': 'false',
    'b64': 'false'
})

print(voice)