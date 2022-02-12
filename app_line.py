import torch
from Config import Config
from tokenizer import Tokenizer
from EncoderDecoder import build_model
from EncoderDecoder import EncoderDecoder
from evaluate import evaluate
import requests

def prep():
    
    id = '1DubBO3fagYccVuKwrIVa4zGvsCzW7FCr'
    destination = ".data/ckpt.pth"
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)
    
    device = torch.device("cpu")

    state_dict = torch.load(f'{Config.data_dir}/{Config.fn}.pth', map_location=device)

    tokenizer = Tokenizer.from_pretrained(Config.model_name)

    model = build_model(Config).to(device)
    model.load_state_dict(state_dict["model"])
    model.eval()
    model.freeze()

def line(message):
    s = message
    print("BOT>", end = "")
    text = evaluate(Config, s, tokenizer, model, device)
    print(text)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)



