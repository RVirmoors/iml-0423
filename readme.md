# iML Art 04.2023 // [bit.ly/iml0423](https://bit.ly/iml0423)

[Grigore Burloiu](https://cinetic.arts.ro/en/echipa/grigore-burloiu/) . [rvirmoors](https://rvirmoors.github.io/) . [ITPMA](https://itpma.notion.site/)

ethics
- [76 reasonable questions](https://76questions.neocities.org/)

---

disclaimer: just some tools!
- no slides[*](https://rvirmoors.github.io/ccia/slides/intro-ml-workshop)[*](https://rvirmoors.github.io/ccia/slides/stylegan-workshop)
- no coding (i hope!)
- no math (almost)
- no art

extra disclaimer: these tools are _fast_
<br/>... but installing them takes TIME and SPACE

0. [How to use this repo](#howto)
1. pose in: [PoseNet](#posenet)
2. audio gen out: [RAVE](#rave)
3. image gen out: [StyleGAN (via FluCoMa)](#stylegan)
4. text out: [GPT4All](#gpt)
5. text to image: [Stable Diffusion](#stablediffusion)

---

## howto

1. clone this repo to your (windows) PC: `git clone https://github.com/RVirmoors/iml-0423 --recursive`
2. open/run each file/script as instructed

see the video recording for details

## posenet

- requires: [Max 8](https://cycling74.com/downloads)
- source: https://github.com/yuichkun/n4m-posenet

open `posenet/hands.maxpat`

## rave

- requires: Max 8
- source: https://github.com/acids-ircam/nn_tilde

open `nn-rave/blabber.maxpat`

## stylegan

- requires: Max 8 with [FluCoMa](https://www.flucoma.org/download/)
- requires: Python 3.9 64bit
- requires: NVidia GPU w/ CUDA drivers
- source: https://github.com/PDillis/stylegan3-fun
- source: https://github.com/marenz2569/Spout-for-Python

[install](stylegan3/startup.txt) StyleGAN3 and trained models

open `stylegan3/pose-gan.maxpat`

open a terminal in `stylegan3/` and run `python gan.py`

## gpt

- requires: Max 8
- requires: Python 3.9 64bit
- source: https://github.com/nomic-ai/gpt4all
- source: https://voicerss.org/sdk/python.aspx (text-to-speech)

edit `configuration copy.yaml` with your [VoiceRSS API key](https://voicerss.org/registration.aspx) and save it as `configuration.yaml`

follow the [instructions](https://github.com/nomic-ai/gpt4all) to download the model .bin file into `gpt4all/chat/`

open a terminal in `gpt-tts/` and run:

```
python -m venv venv
venv/Scripts/activate.bat
pip install pyllamacpp python-osc ffmpeg pyyaml
pyllamacpp-convert-gpt4all ..\gpt4all\chat\gpt4all-lora-quantized.bin .\tokenizer.model ..\gpt4all\chat\gpt4all-lora-converted.bin
python gpt.py
```

open `gpt-tts/haiku.maxpat`

## stablediffusion

- requires: Python 3.9 64bit
- source: https://huggingface.co/stabilityai/stable-diffusion-2-1

again create a venv, install packages, and run the script:

```
cd stable
python -m venv venv
venv/Scripts/activate.bat
pip install light-the-torch
ltt install torch
pip install diffusers transformers accelerate scipy safetensors
python stable.py
```

stable diffusion with stylegan-like interpolation? [why not](https://sites.google.com/view/stylegan-t/)