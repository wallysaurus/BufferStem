# BufferStem
 i try :\

### what is this
its an attempt at live stem separation using a multithreaded-buffer approach.

### why?
i want to make an algorithm for my wled lights at home to be responsive to music, but existing methods (using a microphone and a wave peak algo) suck. using stem separation models to separate percussion elements from the rest of the clutter provides better beat sync.

### how to run
you gotta make a venv like normal. python 3.9 specifically. use the bleeding-edge repo link provided in the demucs readme. the pip library doesn't work. uninstall torch, torchvision, and torchaudio afterwards and instead get the CUDA versions (2.1.2 works best) from the PyTorch website for GPU acceleration (macos not supported). also install sounddevice, and keep uninstalling/reinstalling PySoundFile until it works. Best of luck to you, this takes me about three hours to do myself. SoundDevice doesn't like to play audio on Linux machines so womp womp.
