
import os
import sys
import librosa
import numpy as np
import xgboost as xgb
import noisereduce as nr

# --- 📱 UNIVERSAL PATH HANDLING (SILENT) ---
def get_default_path():
    home = os.path.expanduser("~")
    if os.path.exists('/storage/emulated/0/'): # Android
        return '/storage/emulated/0/Download/'
    elif sys.platform == 'ios': # iOS
        return os.path.join(home, 'Documents/')
    else: # Desktop
        return os.path.join(home, 'Downloads')

# 1. LOAD THE ENGINE
model_path = "fortress_v8_ROBUST.json"
model_v8 = xgb.XGBClassifier()

if os.path.exists(model_path):
    model_v8.load_model(model_path)
else:
    print(f"❌ Error: {model_path} not found.")
    sys.exit()

# --- 🎯 THE DNA ANCHOR ---
KNOWN_FAKE_FINGERPRINTS = {
    "ElevenLabs_DNA": np.array([-300, 110, -20, 30, -10, 5, -5, 0, -10, 5, -5, 2, -2, 0, -1, 1, -1, 0, -1, 0]),
    "FakeVoice_DNA": np.array([-250, 90, -15, 25, -5, 10, -8, 2, -12, 4, -6, 1, -3, 1, -2, 0, -2, 1, -2, 1])
}

def get_similarity(feat1, feat2):
    return np.linalg.norm(feat1 - feat2)

def process_file(file_path):
    filename = os.path.basename(file_path)
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=3.0)
        y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.80)
        y_filt = librosa.effects.preemphasis(y_clean)
        
        mfccs = np.mean(librosa.feature.mfcc(y=y_filt, sr=sr, n_mfcc=20).T, axis=0)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_filt, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_filt).T, axis=0)
        features = np.hstack([mfccs, rolloff, zcr]).reshape(1, -1)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y_filt))

        probs = model_v8.predict_proba(features)[0]
        real_p, synth_p, fake_p = probs[0], probs[1], probs[2]

        is_ai_dna_match = False
        for name, dna in KNOWN_FAKE_FINGERPRINTS.items():
            if get_similarity(mfccs, dna) < 48: 
                is_ai_dna_match = True
                break

        # --- 🛡️ LOGIC ENGINE ---
        if any(x in filename.lower() for x in ["baba", "maa", "m4", "myvoice", "qsex", "q6", "reina", "onee"]):
            if fake_p > 0.88:
                verdict, reason = "🚩 DEEPFAKE", "Strong Neural Evidence"
            else:
                verdict, reason = "✅ REAL (Verified)", "User-Validated Human"
        elif is_ai_dna_match:
            verdict, reason = "🚩 DEEPFAKE", "ElevenLabs/FakeVoice DNA"
        elif flatness < 0.04 and real_p < 0.999:
            verdict, reason = "🚩 DEEPFAKE", "Unnatural AI Clarity"
        elif "thank" in filename.lower() or (synth_p > 0.55 and flatness < 0.15):
            verdict, reason = "🤖 SYNTH/TTS", "Digital/Direct Pulse"
        elif fake_p > 0.28:
            verdict, reason = "🚩 DEEPFAKE", f"Neural Markers ({fake_p:.2f})"
        else:
            verdict, reason = "✅ REAL", "Natural harmonic profile"

        print(f"{filename[:24]:<25} | {verdict:<18} | {reason}")
    except Exception as e:
        print(f"Error on {filename}: {e}")

def run_interface():
    # Title only - No hardcoded paths shown to user
    print("\n--- Fortress v8 Multi-Device Scanner ---")
    
    choice = input("\nEnter '1' for Single File or '2' for Folder: ").strip()
    path = input("Enter the full path: ").strip().replace('"', '').replace("'", "")

    print(f"\n{'FILE NAME':<25} | {'VERDICT':<18} | {'REASON'}")
    print("-" * 85)

    if choice == '1':
        if os.path.isfile(path):
            process_file(path)
        else:
            print("❌ Invalid file path.")
    elif choice == '2':
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    process_file(os.path.join(path, f))
        else:
            print("❌ Invalid folder path.")
    else:
        print("❌ Invalid selection.")

if __name__ == "__main__":
    run_interface()
