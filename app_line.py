import torch
from Config import Config
from tokenizer import Tokenizer
import EncoderDecoder as E_D
from EncoderDecoder import EncoderDecoder
import evaluate as ev

def app_line(message):
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





