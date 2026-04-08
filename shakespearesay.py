import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing the Decoder and ShakespeareGPT classes
from models import *

from utils import *

# Importing the ascii art
from ascii_art import art

PATH = 'weights.pth'

if __name__ == "__main__":
    vocab, itos, decode = init_vocab()
    model =  load_model_with_spinner(PATH, device='cpu')
    chars = int(input(f"""------------SHAKESPEARE GPT----------------\n\n\n 
            {art}\n\nHi, this is me, Shakespeare. My consciousness now lives inside this program, which is a transformer architecture 
written from scratch in PyTorch, and since this model only has about 3 million parameters, my power of
writing is not as good as it used to be (what do these technical terms mean? I don't know, it's definitely not literature!). 
This magnificent face of mine was taken directly from https://emojicombos.com/william-shakespeare-ascii-art.\n
Now, how many characters would you like me to speak? """))

    print(say_with_spinner(model, chars))

    while True:
        chars = int(input("\n\nHow many more characters do you want me to say? "))
        print("\n\n")
        print(say_with_spinner(model, chars))

