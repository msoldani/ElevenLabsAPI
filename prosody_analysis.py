"""
Script per l'analisi prosodica di file audio vocali (.wav).
Estrae parametri fondamentali per la generazione di voce sintetica controllata.

Funzionalità:
- Caricamento file audio
- Estrazione F0 medio, range pitch, durata, velocità eloquio, pause, RMS
- Visualizzazione opzionale di F0 e intensità
- Output in dizionario o JSON

Librerie: parselmouth, librosa, matplotlib, json
"""

import os
import json
import numpy as np
import librosa
import parselmouth
import matplotlib.pyplot as plt
from parselmouth.praat import call
import glob
import csv


def load_audio(file_path, sr=None):
    """
    Carica un file audio .wav.
    Restituisce: y (waveform), sr (sample rate)
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr


def extract_pitch(sound, time_step=0.01, pitch_floor=75, pitch_ceiling=600):
    """
    Estrae la traccia del pitch (F0) usando Parselmouth.
    Restituisce: times, f0_values (Hz)
    """
    pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    pitch_values = pitch.selected_array['frequency']
    times = pitch.xs()
    # Escludi valori zero (non voice)
    f0_values = np.where(pitch_values == 0, np.nan, pitch_values)
    return times, f0_values


def compute_f0_stats(f0_values):
    """
    Calcola F0 medio e range (max-min, ignorando NaN).
    """
    valid_f0 = f0_values[~np.isnan(f0_values)]
    if len(valid_f0) == 0:
        return np.nan, np.nan
    f0_mean = np.mean(valid_f0)
    f0_range = np.max(valid_f0) - np.min(valid_f0)
    return f0_mean, f0_range


def get_duration(sound):
    """
    Restituisce la durata totale dell'audio (secondi).
    """
    return sound.get_total_duration()


def estimate_speech_rate(sound, sr, y):
    """
    Stima la velocità di eloquio (parole/sec) usando Parselmouth (segmentazione silenzi).
    """
    intensity = sound.to_intensity()
    # Calcolo soglia come 20° percentile dei valori validi
    intensity_values = intensity.values[0]
    valid_intensity = intensity_values[~np.isnan(intensity_values)]
    if len(valid_intensity) == 0:
        threshold = 0
    else:
        threshold = np.percentile(valid_intensity, 20)
    min_pause = 0.15  # 150 ms
    pauses = []
    prev_time = 0
    in_pause = False
    for t in np.arange(0, sound.get_total_duration(), 0.01):
        val = intensity.get_value(t)
        if val is not None and val < threshold:
            if not in_pause:
                pause_start = t
                in_pause = True
        else:
            if in_pause:
                pause_end = t
                if pause_end - pause_start > min_pause:
                    pauses.append((pause_start, pause_end))
                in_pause = False
    # Stima parole: approx 1 parola ogni 0.4s di parlato (media italiana)
    total_pause = sum([end - start for start, end in pauses])
    speech_time = sound.get_total_duration() - total_pause
    estimated_words = speech_time / 0.4
    speech_rate = estimated_words / sound.get_total_duration()
    return speech_rate, pauses


def mean_pause_duration(pauses):
    """
    Calcola la durata media delle pause > 150ms.
    """
    if not pauses:
        return 0.0
    durations = [end - start for start, end in pauses]
    return np.mean(durations)


def compute_rms(y):
    """
    Calcola l'RMS medio (in dB) del segnale audio.
    """
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    return np.mean(rms_db)


def compute_jitter_shimmer(sound):
    """
    Calcola jitter e shimmer (facoltativi). Restituisce None se non calcolabili.
    """
    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call([sound, point_process], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        return jitter, shimmer
    except Exception as e:
        return None, None


def plot_pitch(times, f0_values, output_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(times, f0_values, label="F0 (Hz)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Frequenza fondamentale (Hz)")
    plt.title("Andamento F0 nel tempo")
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def plot_rms(y, sr, output_path=None):
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    plt.figure(figsize=(10, 4))
    plt.plot(times, rms, label="RMS")
    plt.xlabel("Tempo (s)")
    plt.ylabel("RMS")
    plt.title("Intensità RMS nel tempo")
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def analyze_prosody(file_path, plot=False, plot_dir=None, compute_jitter_shimmer_flag=False):
    y, sr = load_audio(file_path)
    sound = parselmouth.Sound(file_path)
    times, f0_values = extract_pitch(sound)
    f0_mean, f0_range = compute_f0_stats(f0_values)
    duration = get_duration(sound)
    speech_rate, pauses = estimate_speech_rate(sound, sr, y)
    pause_mean = mean_pause_duration(pauses)
    rms_mean = compute_rms(y)
    result = {
        "f0_mean": float(np.round(f0_mean, 2)) if not np.isnan(f0_mean) else None,
        "f0_range": float(np.round(f0_range, 2)) if not np.isnan(f0_range) else None,
        "duration": float(np.round(duration, 2)),
        "speech_rate": float(np.round(speech_rate, 2)),
        "pause_mean": float(np.round(pause_mean, 2)),
        "rms_mean": float(np.round(rms_mean, 2)),
    }
    if compute_jitter_shimmer_flag:
        jitter, shimmer = compute_jitter_shimmer(sound)
        result["jitter"] = float(np.round(jitter, 5)) if jitter is not None else None
        result["shimmer"] = float(np.round(shimmer, 5)) if shimmer is not None else None
    if plot:
        if not plot_dir:
            plot_pitch(times, f0_values)
            plot_rms(y, sr)
        else:
            os.makedirs(plot_dir, exist_ok=True)
            plot_pitch(times, f0_values, os.path.join(plot_dir, "f0_plot.png"))
            plot_rms(y, sr, os.path.join(plot_dir, "rms_plot.png"))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Estrazione parametri prosodici da file audio .wav")
    parser.add_argument("file", nargs="?", help="Percorso file .wav (ignorato se usi --batch)")
    parser.add_argument("--plot", action="store_true", help="Mostra grafici F0 e RMS")
    parser.add_argument("--plot-dir", type=str, default=None, help="Directory per salvare i grafici")
    parser.add_argument("--jitter-shimmer", action="store_true", help="Calcola anche jitter e shimmer")
    parser.add_argument("--json", type=str, default=None, help="Salva output in file JSON (solo modalità singolo file)")
    parser.add_argument("--batch", type=str, default=None, help="Analizza tutti i .wav in questa cartella")
    parser.add_argument("--csv", type=str, default=None, help="Salva risultati multipli in CSV (solo con --batch)")
    args = parser.parse_args()

    if args.batch:
        # Analisi batch su tutti i .wav nella cartella
        wav_files = sorted(glob.glob(os.path.join(args.batch, "*.wav")))
        results = []
        for wav in wav_files:
            print(f"Analizzo: {os.path.basename(wav)}")
            res = analyze_prosody(wav, plot=False, plot_dir=None, compute_jitter_shimmer_flag=args.jitter_shimmer)
            res["filename"] = os.path.basename(wav)
            results.append(res)
        if args.csv:
            # Scrivi CSV
            fieldnames = [
                "filename", "f0_mean", "f0_range", "duration", "speech_rate", "pause_mean", "rms_mean", "jitter", "shimmer"
            ]
            with open(args.csv, "w", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    # Assicura che tutte le colonne siano presenti
                    for k in fieldnames:
                        if k not in row:
                            row[k] = None
                    writer.writerow(row)
            print(f"Risultati salvati in {args.csv}")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        # Analisi singolo file
        if not args.file:
            print("Errore: specifica un file .wav o usa --batch")
            return
        result = analyze_prosody(args.file, plot=args.plot, plot_dir=args.plot_dir, compute_jitter_shimmer_flag=args.jitter_shimmer)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main() 