import argparse, torch, torchaudio, faiss, pickle, os
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram
import pickle
import numpy as np
from tqdm import tqdm
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
    
def multi_segment_audio_processing(directory_path, speaker_id):
    mel_tensors = []  # List to store mel tensors for all segments

    # Iterate over all audio files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.wav') and file_name.startswith('20170001P00' + speaker_id):
            file_path = os.path.join(directory_path, file_name)

            # Process the audio and obtain the mel tensor
            wav_tensor, sample_rate = torchaudio.load(file_path)
            mel_tensor = wav2mel(wav_tensor, sample_rate)
            mel_tensors.append(mel_tensor)
    # Use dvector.embed_utterances to convert mel tensors of multiple segments
    emb_tensor = dvector.embed_utterances(mel_tensors)  # shape: (emb_dim)

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

    # Get a list of all speaker IDs
    speaker_ids = set()
    file_list = [file_name for file_name in os.listdir(directory_path) if file_name.endswith('.wav')]
    for file_name in file_list:
        speaker_id = file_name[-12:-8]  # Extract the four-digit ID from the file name
        speaker_ids.add(speaker_id)
    #print(speaker_ids)
    # Loop over speaker IDs with a progress bar
    progress_bar = tqdm(speaker_ids, desc="Processing speakers", unit="speaker")
    for speaker_id in progress_bar:
        # Process multiple segments of the same speaker
        emb_tensor = multi_segment_audio_processing(directory_path, speaker_id)
        emb_numpy = emb_tensor.detach().numpy().reshape(1, -1)

        # Add the new vector to the index
        index.add(emb_numpy)
        # Add the name to the list
        names.append(speaker_id)

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
