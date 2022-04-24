# Active-DSP-Speaker-Crossover

This was made for a class project. It is functional, but still needs the following to be improved:
- Using multiple audio outputs at once can result in a crash
- There is popping when the output streams switch between "Chunks" of audio
- There is no command line control
- File paths are not relative
- There is no volume control beyond the scaling of the Chunk data

Dispite these problems, the system does work. Do use it, you must change the following to match your system.
- The input and output speaker device numbers must be changed. You can find the correct numbers by running the following command: ```python -m sounddevice```
- The file paths to any audio files must be changed to match your device
