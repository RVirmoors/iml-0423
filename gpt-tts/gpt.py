from pyllamacpp.model import Model
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from random import random

ip = "127.0.0.1"
in_port = 1337
out_port = 12000
client = SimpleUDPClient(ip, out_port)  # Create client

def prompt_handler(address, *args):
    width = args[0]
    height = args[1]
    thing = "any one thing" if random() < 0.5 else "an animal"
    if width > height:
        prompt = "Name " + thing + " that is around " + str(width) + " cm wide."
    else:
        prompt = "Name " + thing + " that is very tall and write a haiku about it." # around " + str(height) + " cm tall."
    print(prompt)
    nlines = 0
    haiku = ''
    while nlines < 3:
        generated_text = model.generate(prompt, n_predict=100)
        print("AAA ", generated_text)
        haiku = " ".join(generated_text.split(".")[1:])
        print("BBB ", haiku)
        nlines = haiku.count('\n')

    print("HAIKU: ", haiku.lstrip())
    client.send_message("/haiku", haiku.lstrip())


dispatcher = Dispatcher()
dispatcher.map("/prompt", prompt_handler)

model = Model(ggml_model='..\gpt4all\chat\gpt4all-lora-converted.bin', n_ctx=512)

server = BlockingOSCUDPServer((ip, in_port), dispatcher)
server.serve_forever()  # Blocks forever