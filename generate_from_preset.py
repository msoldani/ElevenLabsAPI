import json
import sys
from pathlib import Path
from datetime import datetime

def generate_audio(text, preset_file='donolato_preset.json'):
    # Carica preset
    with open(preset_file) as f:
        preset = json.load(f)

    # Applica il template SSML
    ssml = preset["ssml_template"].format(text=text)

    # Stampa info
    print(f"\nGenerando audio per il testo:\n{text}")
    
    # Usa la funzione text_to_speech di ElevenLabs
    from mcp_ElevenLabs_text_to_speech import text_to_speech
    
    result = text_to_speech(
        text=ssml,
        voice_id=preset["voice_id"],
        output_format="mp3_44100_128",
        stability=preset["stability"],
        similarity_boost=preset["similarity_boost"],
        speed=preset["speed"],
        style=preset["style"]
    )
    
    print(f"\nAudio generato: {result}")
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 generate_from_preset.py \"testo1\" \"testo2\" \"testo3\" ...")
        print("Esempio: python3 generate_from_preset.py \"Offerta speciale!\" \"Nuovo arrivo!\"")
        sys.exit(1)
    
    # Genera audio per ogni testo fornito come argomento
    for text in sys.argv[1:]:
        generate_audio(text) 
