import sys
sys.path.append('stylegan3-fun')
# syphon for Apple Metal texture sending
import syphon
from syphon.utils.raw import create_mtl_texture
from syphon.utils.numpy import copy_image_to_mtl_texture

import cv2

# ---- SG imports ---
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import torch

import legacy

network_pkl = "stylegan3-r-ffhqu-256x256.pkl"

# ---- OSC -----
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

def get_psi(address, *args):
    global psi
    # print("PSI:", args[0])
    psi = args[0]

def get_z(address, *args):
    global z
    z = torch.from_numpy(np.array(args).reshape(1, 512).astype(np.float32)).to(device)

dispatcher = Dispatcher()
dispatcher.map("/psi", get_psi)
dispatcher.map("/z", get_z)

osc_server = BlockingOSCUDPServer(("localhost", 5555), dispatcher)

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
    global syphon_server, texture, device, G, psi, z
    # create spout object
    syphon_server = syphon.SyphonMetalServer('output')
    texture = create_mtl_texture(syphon_server.device, 256, 256)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('mps')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    psi = 1
    seed = 0
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim).astype(np.float32)).to(device)
    # print(G)

def update():
    global syphon_server, texture, device, G, psi, z
    # print("Generate image.")
    osc_server.handle_request() # get new OSC

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
    alpha_channel = torch.full((1, 256, 256, 1), 255, dtype=img.dtype, device=img.device)
    img = torch.cat((img, alpha_channel), dim=-1)
    # print(img.shape)
    data = img[0].cpu().numpy()
    # print(data.shape)
    # send data
    # print(data[0][0][0])
    cv2.imshow("output", data)
    cv2.waitKey(1)
    copy_image_to_mtl_texture(data, texture)
    syphon_server.publish_frame_texture(texture)

#-------------
setup()
while True: # Ctrl+C to stop
    update()
