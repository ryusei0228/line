import torch
from mylib from Config import Config
from mylib from tokenizer import Tokenizer
from mylib import EncoderDecoder as E_D
from mylib from EncoderDecoder import EncoderDecoder
from mylib import evaluate as ev

def line(message):
    device = torch.device("cpu")

    state_dict = torch.load(f'{Config.data_dir}/{Config.fn}.pth', map_location=device)

    tokenizer = Tokenizer.from_pretrained(Config.model_name)

    model = E_D.build_model(Config).to(device)
    model.load_state_dict(state_dict["model"])
    model.eval()
    model.freeze()

    while True:
        s = message
        if s == "q":
           break
        print("BOT>", end = "")
        text = ev.evaluate(Config, s, tokenizer, model, device)
        print(text)





