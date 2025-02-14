import torch
from flask import Flask, render_template, request
from torch import nn
from pyvi import ViTokenizer
import spacy
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from model import Seq2Seq, load_model, tokenize_en, tokenize_vi
from datasets import load_dataset
from torchtext.vocab import Vocab
from collections import Counter

# Load model and tokenizer
spacy_en = spacy.load('en_core_web_sm')
model_path = "../task1+2+3/general_attention_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = Seq2Seq(input_dim=10000, output_dim=10000, hidden_size=512, attention_method="general")
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Load dataset and vocab for tokenization
SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'vi'
dataset = load_dataset('opus100', f'{SRC_LANGUAGE}-{TRG_LANGUAGE}')
train = dataset['train']

# Tokenization functions
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_vi(text):
    return ViTokenizer.tokenize(text).split()

# Define token transform dictionary
token_transform = {
    SRC_LANGUAGE: tokenize_en,
    TRG_LANGUAGE: tokenize_vi
}

# Helper function to yield list of tokens
def yield_tokens(data, language):
    language_index = {SRC_LANGUAGE: 'en', TRG_LANGUAGE: 'vi'}
    for data_sample in data:
        yield token_transform[language](data_sample['translation'][language_index[language]])

# Build vocabulary
vocab_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    # Count token occurrences
    counter = Counter()
    for tokens in yield_tokens(train, ln):
        counter.update(tokens)

    # Create the vocab and add special tokens manually
    vocab = Vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], specials_first=True)

    # Set the default index (e.g., for unknown words)
    vocab_transform[ln] = vocab

def translate(sentence):
    # Tokenize source sentence (English)
    src = tokenize_en(sentence)
    src_tensor = torch.tensor([vocab_transform['en'][word] for word in src]).unsqueeze(0).to(device)  # Add batch dimension
    trg_tensor = torch.zeros(1, len(src), dtype=torch.long).to(device)  # Assuming target size is same as source
    with torch.no_grad():
        output = model(src_tensor, trg_tensor)
        translated_sentence = ' '.join([vocab_transform['vi'].lookup_token(int(word)) for word in output.squeeze(0)])
    return translated_sentence

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    translation = ""
    if request.method == "POST":
        sentence = request.form["sentence"]
        translation = translate(sentence)
    return render_template("index.html", translation=translation)

if __name__ == "__main__":
    app.run(debug=True)