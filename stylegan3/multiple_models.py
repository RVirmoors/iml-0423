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

#network_pkl1 = "G:/My Drive/CC3_stylegan/stylegan3-fun/training-runs/00018-stylegan3-t-shadow-1024x1024-gpus1-batch16-gamma32-resume_custom/network-snapshot-000052.pkl"
#network_pkl2 = "G:/My Drive/CC3_stylegan/stylegan3-fun/training-runs/00022-stylegan3-t-self-1024x1024-gpus1-batch16-gamma32-resume_custom/network-snapshot-000100.pkl"

network_pkl = ["G:/My Drive/CC3_stylegan/stylegan3-fun/training-runs/00018-stylegan3-t-shadow-1024x1024-gpus1-batch16-gamma32-resume_custom/network-snapshot-000052.pkl", 
               "G:\My Drive\CC3_stylegan\stylegan3-fun/training-runs/00020-stylegan3-t-anima-1024x1024-gpus1-batch16-gamma32-resume_custom/network-snapshot-000064.pkl", 
               "G:\My Drive\CC3_stylegan\stylegan3-fun/training-runs/00019-stylegan3-t-animus-1024x1024-gpus1-batch16-gamma32-resume_custom/network-snapshot-000048.pkl",
               "G:\My Drive\CC3_stylegan\stylegan3-fun/training-runs/00022-stylegan3-t-self-1024x1024-gpus1-batch16-gamma32-resume_custom/network-snapshot-000100.pkl",
               "G:\My Drive\CC3_stylegan\stylegan3-fun/training-runs/00021-stylegan3-t-persona-1024x1024-gpus1-batch16-gamma32-resume_custom/network-snapshot-000100.pkl"
               ]

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

def switch(address, *args):
    global archetype
    print("archetype:", args[0])
    archetype = args[0]

def get_seed(address, *args):
    global z, seed
    # print("seed:", args[0])
    seed = args[0]
    z = z + torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(device)

dispatcher = Dispatcher()
dispatcher.map("/archetype", switch)
dispatcher.map("/psi", get_psi)
dispatcher.map("/z", get_z)
dispatcher.map("/seed", get_seed)

server = BlockingOSCUDPServer(("localhost", 5555), dispatcher)

#-------------
translate = (0,0)
rotate = 0
archetype = 0 
seed = 0
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

    #print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    G = [0, 0, 0, 0, 0]
    for i in range (5):
        with dnnlib.util.open_url(network_pkl[i]) as f:
            G[i] = legacy.load_network_pkl(f)['G_ema'].to(device)
            #print (G[i])
    psi = 1
    seed = 0
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G[0].z_dim)).to(device)
    # print(G)

def update():
    global spout, device, G, psi, z, seed, archetype
    # print("Generate image.")
    server.handle_request() # get new OSC
    # check on close window
    spout.check()

    label = torch.zeros([1, 512], device=device)
    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    # if hasattr(G.synthesis, 'input'):
        # m = make_transform(translate, rotate)
        # m = np.linalg.inv(m)
        # G.synthesis.input.transform.copy_(torch.from_numpy(m))

    g=G[archetype]
    img = g(z, label, truncation_psi=psi, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # print(img.shape)
    data = img[0].cpu().numpy()
    print (archetype, seed)
    # send data
    spout.send(data)

setup()
while True: # Ctrl+C to stop
    update()
