import sys
sys.path.append('stylegan3-fun')
# load library
from Library.Spout import Spout

# ---- SG imports ---
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import torch

import legacy

network_pkl = "G:\My Drive/CC3_stylegan/stylegan3-fun/training-runs/00018-stylegan3-t-shadow-1024x1024-gpus1-batch16-gamma32-resume_custom/network-snapshot-000052.pkl"

# ---- OSC -----
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

def get_psi(address, *args):
    global psi
    # print("PSI:", args[0])
    psi = args[0]

def get_z(address, *args):
    global z
    z = torch.from_numpy(np.array(args).reshape(1, 512)).to(device)

dispatcher = Dispatcher()
dispatcher.map("/psi", get_psi)
dispatcher.map("/z", get_z)

server = BlockingOSCUDPServer(("localhost", 5555), dispatcher)

#-------------
translate = (0,0)
rotate = 0
def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#-------------
def setup():
    global spout, device, G, psi, z
    # create spout object
    spout = Spout(silent = False, width = 1024, height = 1024)
    # create sender
    spout.createSender('output')

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    psi = 1
    seed = 0
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    # print(G)

def update():
    global spout, device, G, psi, z
    # print("Generate image.")
    server.handle_request() # get new OSC
    # check on close window
    spout.check()

    label = torch.zeros([1, G.c_dim], device=device)
    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    # if hasattr(G.synthesis, 'input'):
        # m = make_transform(translate, rotate)
        # m = np.linalg.inv(m)
        # G.synthesis.input.transform.copy_(torch.from_numpy(m))

    img = G(z, label, truncation_psi=psi, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # print(img.shape)
    data = img[0].cpu().numpy()
    # send data
    spout.send(data)

setup()
while True: # Ctrl+C to stop
    update()
