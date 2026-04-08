import torch
import threading
import sys
import time
from models import ShakespeareGPT

BLOCK_SIZE = 256
NUM_BLOCKS = 6

spinner_text = None

def init_vocab():
    with open('input.txt') as f:
            text = f.read()

    vocab = sorted(list(set(text)))
    itos = {i: s for i, s in enumerate(vocab)}
    decode = lambda x: ''.join([itos[i] for i in x])
    return vocab, itos, decode

def spinner_task(stop_event):
    """Runs the spinner in a separate thread until stop_event is set."""
    global spinner_text
    chars = ['|', '/', '-', '\\']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f'\r{spinner_text[0]}... {chars[i % len(chars)]}')
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    #sys.stdout.write(f'\r{spinner_text[1]}\n')
    #sys.stdout.flush()
    print(f"\n{spinner_text[1]}\n")

def load_model_with_spinner(path, device='cpu'):
    global spinner_text
    spinner_text = ["Loading model", "Model loaded successfully! ✓"]
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner_task, args=(stop_event,))

    spinner_thread.start()
    try:
        model = ShakespeareGPT(T=BLOCK_SIZE, h=4, num_blocks=NUM_BLOCKS)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    finally:
        stop_event.set()       # Stop the spinner whether or not loading succeeded
        spinner_thread.join()  # Wait for the spinner thread to finish cleanly

    return model



def say_with_spinner(model, max_new_tokens):
    global spinner_text
    spinner_text = ["Hold up I'm thinking", "Aha! Here you go\n"]
    vocab, itos, decode = init_vocab()
    context = torch.zeros((1, 1), dtype=torch.long, device='cpu')

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner_task, args=(stop_event,))

    spinner_thread.start()

    try:
        to_return = decode(model.generate(context, max_new_tokens)[0].tolist())
    finally:
        stop_event.set()
        spinner_thread.join()

    return to_return



