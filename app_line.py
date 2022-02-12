import torch
from mylib.Config import Config
from mylib.tokenizer import Tokenizer
from mylib.EncoderDecoder import build_model
from mylib.EncoderDecoder import EncoderDecoder
from mylib.evaluate import evaluate

def line(message):
    device = torch.device("cpu")

    state_dict = torch.load(f'{Config.data_dir}/{Config.fn}.pth', map_location=device)

    tokenizer = Tokenizer.from_pretrained(Config.model_name)

    model = build_model(Config).to(device)
    model.load_state_dict(state_dict["model"])
    model.eval()
    model.freeze()

    while True:
        s = message
        if s == "q":
           break
        print("BOT>", end = "")
        text = evaluate(Config, s, tokenizer, model, device)
        print(text)





