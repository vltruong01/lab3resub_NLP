import torch
import torch.nn as nn
import torch.nn.functional as F
from pyvi import ViTokenizer
import spacy

# Load English tokenizer
spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_vi(text):
    return ViTokenizer.tokenize(text).split()

class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        # For multiplicative attention
        if method == "multiplicative":
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        # For additive attention
        elif method == "additive":
            self.W1 = nn.Linear(hidden_size, hidden_size)
            self.W2 = nn.Linear(hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        if self.method == "general":
            # General attention (dot product attention)
            attn_energies = torch.sum(hidden * encoder_outputs, dim=2)  # (batch, seq_len)
        elif self.method == "multiplicative":
            # Multiplicative attention (learned linear transformation before dot product)
            attn_energies = torch.sum(hidden * self.W(encoder_outputs), dim=2)  # (batch, seq_len)
        elif self.method == "additive":
            # Additive attention (MLP based)
            attn_energies = self.v(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))  # (batch, seq_len, 1)
            attn_energies = attn_energies.squeeze(2)  # (batch, seq_len)
        
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # (batch, 1, seq_len)

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, attention_method):
        super(Seq2Seq, self).__init__()
        
        # Encoder: LSTM layer
        self.encoder = nn.LSTM(input_dim, hidden_size, batch_first=True)
        
        # Decoder: LSTM layer
        self.decoder = nn.LSTM(output_dim, hidden_size, batch_first=True)
        
        # Attention mechanism (pass it to the constructor)
        self.attention = Attention(attention_method, hidden_size)
        
        # Fully connected layer to project to the output vocab space
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, src, trg):
        # Pass source through the encoder
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Pass target through the decoder
        decoder_outputs, _ = self.decoder(trg, (hidden, cell))
        
        # Calculate attention weights based on the decoder outputs
        attn_weights = self.attention(decoder_outputs, encoder_outputs)
        
        # Calculate context vector (weighted sum of encoder outputs)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden_size)

        # Pass context through fully connected layer
        output = self.fc(context)  # (batch, 1, output_dim)
        
        return output

# Load the pre-trained model from the file
def load_model(model_path):
    input_dim = 10000  # Size of the source vocabulary
    output_dim = 10000  # Size of the target vocabulary
    hidden_size = 512  # Hidden size for LSTM
    model = Seq2Seq(input_dim, output_dim, hidden_size, "general")  # Initialize model
    model.load_state_dict(torch.load(model_path))  # Load trained weights
    model.eval()  # Set to evaluation mode
    return model