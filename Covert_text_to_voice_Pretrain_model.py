#!/usr/bin/env python3
from TTS.api import TTS
import os
import sys
from pathlib import Path
import subprocess
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Add safe globals for PyTorch 2.6 compatibility
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Configuration
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
OUTPUT_DIR = os.path.expanduser("~/TTS_output")
REFERENCE_VOICE = "/home/nathan/TTS/Ref/Trump_ref2.wav"

def load_tts_model():
    """Load TTS model with enhanced error handling"""
    from TTS.utils.manage import ModelManager
    model_manager = ModelManager()
    
    # Download model (no force_redownload argument)
    model_path, config_path, _ = model_manager.download_model(MODEL_NAME)
    
    if not model_path or not config_path:
        raise ValueError("❌ Failed to download model - check network connection")
    
    try:
        return TTS(model_path=model_path, config_path=config_path, gpu=False)
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        # Try with explicit config
        config = XttsConfig()
        config.load_json(config_path)
        return TTS.init_from_config(config)
    

def convert_file(input_path, output_dir=OUTPUT_DIR, play_audio=False):
    """Convert text file to speech"""
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load TTS model
        tts = load_tts_model()
        print("✅ Model loaded successfully")
        
        # Read input text
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if not text:
                raise ValueError("Empty input file")
        
        # Set output path
        input_name = Path(input_path).stem
        output_path = f"{output_dir}/{input_name}.wav"
        
        # Check reference voice
        use_reference = os.path.exists(REFERENCE_VOICE)
        if not use_reference:
            print("⚠️ Reference voice not found, using default voice")
        
        # Convert text to speech
        print(f"🔊 Converting '{Path(input_path).name}' to English...")
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=REFERENCE_VOICE if use_reference else None,
            language="en"
        )
        
        # Notification and playback
        subprocess.run(["notify-send", "TTS Conversion Complete", f"Audio saved to {output_path}"])
        if play_audio:
            subprocess.run(["aplay", "-q", output_path])
        
        print(f"✅ Success: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        subprocess.run(["notify-send", "TTS Error", str(e)])
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.txt> [output_dir] [--play]")
        print(f"Example: {sys.argv[0]} document.txt ~/audio_output --play")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    play_audio = "--play" in sys.argv
    
    convert_file(input_path, output_dir, play_audio)