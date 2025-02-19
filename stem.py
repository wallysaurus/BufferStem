import asyncio
import queue
import threading
import concurrent.futures
import sounddevice as sd
import demucs.api
import torchaudio as ta
import torch
import math
import time
import numpy as np

# Load the audio file
wav, sr = ta.load("loyalty.wav")
# wav = torch.tensor(
#         [
#             wav[0].tolist()[: sr * 30],
#             wav[1].tolist()[: sr * 30]
#         ]
#     )

separator = demucs.api.Separator(progress=True, jobs=5, overlap=0.1, device="cuda")

# This function runs on its own thread, so it's okay that we
# shove all the chunks immediately into the separator. Once
# each chunk finishes, it'll get passed to the queue and played
# on another thread giving the illusion of asynchronous output.
def stemsplit(q, duration=10, output="vocals"):
    num_chunks = math.floor(len(wav[0]) / (sr * duration))
    for i in range(num_chunks):
        three_seconds = torch.tensor(
            [
                # Sample Rate * iteration offset * duration (seconds) = startpos
                wav[0].tolist()[sr * i * duration:][: sr * duration],
                wav[1].tolist()[sr * i * duration:][: sr * duration]
            ]
        )
        _, separated = separator.separate_tensor(three_seconds)
        q.put(separated[output])  # Add the separated chunk to the queue

# This will eventually get cleaned up. It sucks right now.
def play_audio(q):
    current_chunk = None  # Holds the current audio chunk being processed
    pos = 0  # Current position within the chunk

    def callback(outdata, frames, time, status):
        nonlocal current_chunk, pos
        if status:
            print(f"Underflow: {status}", file=sys.stderr)

        # Fetch new chunk if current is exhausted
        if current_chunk is None or pos >= current_chunk.shape[0]:
            try:
                chunk = q.get_nowait()
                if chunk is None:
                    raise sd.CallbackAbort
                # Convert tensor to numpy and transpose to (samples, channels)
                current_chunk = chunk.numpy().T.astype(np.float32)
                pos = 0
            except queue.Empty:
                # Temporarily fill with zeros if queue is empty
                outdata.fill(0)
                return

        # Calculate how much data we can copy
        remaining = current_chunk.shape[0] - pos
        if remaining == 0:
            current_chunk = None
            outdata.fill(0)
            return

        # Copy data to output buffer
        to_copy = min(frames, remaining)
        outdata[:to_copy] = current_chunk[pos:pos + to_copy]
        pos += to_copy

        # Pad with zeros if we couldn't fill the buffer
        if to_copy < frames:
            outdata[to_copy:] = 0

    # Configure and start the output stream
    with sd.OutputStream(
        samplerate=sr,
        channels=2,
        callback=callback,
        blocksize=1024  # Lower latency (adjust based on your needs)
    ):
        print("Playback started")
        # Keep the stream alive until termination signal is received
        while True:
            sd.sleep(1000)  # Prevent main thread from exiting

    print("Playback finished")

def main():
    q = queue.Queue()  # Thread-safe queue for audio chunks

    # Create and start threads
    separation_thread = threading.Thread(target=stemsplit, args=(q,))
    playback_thread = threading.Thread(target=play_audio, args=(q,))

    playback_thread.start()
    separation_thread.start()

    # Wait for the separation thread to finish
    separation_thread.join()
    # Signal the playback thread to exit
    q.put(None)
    # Wait for the playback thread to finish
    playback_thread.join()

if __name__ == "__main__":
    main()
