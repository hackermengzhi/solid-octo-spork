import wave
import os


original_file = wave.open("example.wav", "rb")


params = original_file.getparams()
num_channels = params.nchannels
sample_width = params.sampwidth
frame_rate = params.framerate
num_frames = params.nframes


segment_size = 50000


total_segments = num_frames // segment_size


os.makedirs("segments", exist_ok=True)


for segment_index in range(total_segments):

    start_frame = segment_index * segment_size
    end_frame = (segment_index + 1) * segment_size


    original_file.setpos(start_frame)


    segment_frames = original_file.readframes(segment_size)


    segment_file_name = f"segments/example_{segment_index}.wav"
    segment_file = wave.open(segment_file_name, "wb")


    segment_file.setparams(params)


    segment_file.writeframes(segment_frames)


    segment_file.close()


    os.system(f"python3 demo.py --file {segment_file_name}")


original_file.close()
