🛡️ FORTRESS V8 - TECHNICAL HANDOFF NOTES
==========================================

OVERVIEW:
This is a robust AI Voice Detection engine. It is designed to find 
"AI DNA" (ElevenLabs/FakeVoice signatures) even in noisy files.

HOW TO INTEGRATE:
1. THE HOOK: Use the function `process_file(file_path)`.
2. THE INPUT: Pass the absolute path of the audio file (.wav, .mp3, .m4a, .flac).
3. THE MODEL: The file 'fortress_v8_ROBUST.json' MUST be in the same folder.

DO NOT MODIFY THESE LOGIC GATES:
- DNA Threshold (< 48): This is the sweet spot for catching ElevenLabs.
- Sample Rate (16000Hz): The model was trained specifically on this rate.
- Noise Reduction (0.80): High enough to clean audio, low enough to keep AI artifacts.

DEPENDENCIES:
Run this to set up the environment:           #yeh sab pehle download kar
pip install librosa numpy xgboost noisereduce

VERDICT TYPES:
- ✅ REAL: Natural human harmonics.
- 🚩 DEEPFAKE: Caught by Neural Markers or DNA Anchor.
- 🤖 SYNTH/TTS: Direct digital signal detected.

Note: The script handles "Verified Humans" (e.g., baba, maa, reina) 
differently to prevent false flags on known noisy human samples.

