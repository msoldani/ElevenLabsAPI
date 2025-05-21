import parselmouth
import numpy as np
import os
import csv

def analyze_prosody(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    intensity = snd.to_intensity()
    
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    min_pitch = np.min(pitch_values) if len(pitch_values) > 0 else 0
    max_pitch = np.max(pitch_values) if len(pitch_values) > 0 else 0
    
    intensity_values = intensity.values[0]
    avg_intensity = np.mean(intensity_values)
    min_intensity = np.min(intensity_values)
    max_intensity = np.max(intensity_values)
    
    total_duration = snd.get_total_duration()
    silence_threshold = 0.01 * max_intensity
    pauses = np.where(intensity_values < silence_threshold)[0]
    pause_ratio = len(pauses) / len(intensity_values)
    approx_speech_rate = (len(pitch_values) / total_duration) / 5 if total_duration > 0 else 0
    
    return {
        "avg_pitch": avg_pitch,
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "avg_intensity": avg_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "total_duration": total_duration,
        "pause_ratio": pause_ratio,
        "approx_speech_rate": approx_speech_rate
    }

# Percorso della cartella con i file audio
audio_folder = "Materiali"
output_csv = "report_prosodia_demarco.csv"
write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0

with open(output_csv, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Scrivi l'intestazione solo se il file è nuovo o vuoto
    if write_header:
        writer.writerow([
            "file", "avg_pitch", "min_pitch", "max_pitch",
            "avg_intensity", "min_intensity", "max_intensity",
            "total_duration", "pause_ratio", "approx_speech_rate"
        ])
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav"):
            path = os.path.join(audio_folder, filename)
            data = analyze_prosody(path)
            writer.writerow([
                filename,
                data["avg_pitch"], data["min_pitch"], data["max_pitch"],
                data["avg_intensity"], data["min_intensity"], data["max_intensity"],
                data["total_duration"], data["pause_ratio"], data["approx_speech_rate"]
            ])

print(f"Report generato: {output_csv}")
