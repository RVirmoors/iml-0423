from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

ip = "127.0.0.1"
in_port = 6448

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-1"

def haiku_handler(address, *args):
    haiku = args[0]
    print(haiku)
    image = pipe(haiku).images[0]
        
    image.save("../gpt-tts/haiku.png")
    print(image)

dispatcher = Dispatcher()
dispatcher.map("/haiku", haiku_handler)

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

server = BlockingOSCUDPServer((ip, in_port), dispatcher)
server.serve_forever()  # Blocks forever