import argparse, torch, torchaudio, faiss, pickle, os
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram
import pickle
import numpy as np

# Load the model
wav2mel = torch.jit.load("wav2mel.pt")
dvector = torch.jit.load("dvector.pt").eval()
id_to_name={}

def audio_procession(file_path):
    # Load the waveform
    wav_tensor, sample_rate = torchaudio.load(file_path)
    mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)

    # Compute embedding
    emb_tensor = dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
    return emb_tensor

def registration_P(directory_path):
    index_file = "emb_index.index"
    if os.path.exists(index_file):
        # Load the existing index
        index = faiss.read_index(index_file)
    else:
        # Create a new index
        index = faiss.IndexFlatL2(256)  # Assuming the dimension of your embeddings is 256

    names = []  # List to store the names

    # Loop over all files in the directory
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        name = os.path.splitext(os.path.basename(file_path))[0]
        emb_tensor = audio_procession(file_path)
        emb_numpy = emb_tensor.detach().numpy().reshape(1, -1)

        # Add the new vector to the index
        index.add(emb_numpy)

        # Add the name to the list
        names.append(name)

    # Save the names list to a file
    with open('names.pkl', 'wb') as f:
        pickle.dump(names, f)

    # Write the index back to the file
    faiss.write_index(index, index_file)


def recognition(file_path):
    # Load the names list from the file
    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)

    emb_tensor = audio_procession(file_path)
    emb_numpy = emb_tensor.detach().numpy().reshape(1, -1)
    index_file = "emb_index.index"
    index = faiss.read_index(index_file)
    k = 1
    D, I = index.search(emb_numpy, k)
    print(I)
    # Look up the names for the returned IDs
    returned_names = [names[i] for i in I[0]]

    print(returned_names[0])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo script")
    parser.add_argument("--file", type=str, help="Input file path")
    parser.add_argument("--mode", type=str, help="Mode: register or recognize")
    args = parser.parse_args()
    
    if args.mode == "register":
        registration_P(args.file)
    elif args.mode == "recognize":
        recognition(args.file)
