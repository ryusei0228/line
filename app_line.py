import torch
from Config import Config
from tokenizer import Tokenizer
from EncoderDecoder import build_model
from EncoderDecoder import EncoderDecoder
from evaluate import evaluate
import requests

def prep():
       
    device = torch.device("cpu")

    state_dict = torch.load(f'data/ckpt.pth', map_location=device)

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




