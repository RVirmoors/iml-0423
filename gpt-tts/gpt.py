from pyllamacpp.model import Model
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from random import random

import yaml
import voicerss_tts
import wave

with open('configuration.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

ip = "127.0.0.1"
in_port = 1337
out_port = 12000
client = SimpleUDPClient(ip, out_port)  # Create client

def say(what):
    voice = voicerss_tts.speech({
        'key': config['api_key'],
        'hl': config['language'],
        'v': config['voice'],
        'src': what,
        'r': config['speed'],
        'c': config['codec'],
        'f': config['format'],
        'ssml': 'false',
        'b64': 'false'
    })

    with wave.open('output.wav', 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(44100)
        wav_file.writeframes(voice['response'])


def prompt_handler(address, *args):
    width = args[0]
    height = args[1]
    thing = "any one thing" if random() < 0.5 else "an animal"
    if width > height:
        if width > 300:
            howbig = "immensely wide"
        elif width > 200:
            howbig = "very wide"
        elif width > 122:
            howbig = "as wide as a door"
        elif width > 50:
            howbig = "slender"
        else:
            howbig = "very small"
        prompt = "Name " + thing + " that is " + howbig + " and write a haiku about it."
    else:
        if height > 300:
            howbig = "hugely tall"
        elif height > 200:
            howbig = "very tall"
        elif height > 122:
            howbig = "as tall as a person"
        elif height > 50:
            howbig = "smaller than a person"
        else:
            howbig = "very small"
        prompt = "Name " + thing + " that is " + howbig + " and write a haiku about it."
    print("PROMPT:", prompt)

    generated_text = model.generate(prompt, n_predict=100)
    # print("AAA ", generated_text)
    haiku = " ".join(generated_text.split(".")[1:])
    # print("BBB ", haiku)
    nlines = haiku.count('\n')

    if nlines < 3:
        client.send_message("/again", 0)
        return
    else:
        print("HAIKU:", haiku.lstrip())
        say(haiku.lstrip())
        client.send_message("/haiku", haiku.lstrip())


dispatcher = Dispatcher()
dispatcher.map("/prompt", prompt_handler)

model = Model(ggml_model='..\gpt4all\chat\gpt4all-lora-converted.bin', n_ctx=512)

server = BlockingOSCUDPServer((ip, in_port), dispatcher)
server.serve_forever()  # Blocks forever