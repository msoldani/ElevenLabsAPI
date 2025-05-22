import os
import requests
import gradio as gr

ELEVEN_BASE_URL = "https://api.elevenlabs.io"

def headers(api_key: str):
    return {"xi-api-key": api_key.strip()}


def fetch_voices(api_key: str):
    """Return dict {name: voice_id} from ElevenLabs API."""
    resp = requests.get(f"{ELEVEN_BASE_URL}/v1/voices", headers=headers(api_key))
    resp.raise_for_status()
    data = resp.json()
    return {v["name"]: v["voice_id"] for v in data.get("voices", [])}


def text_to_speech(api_key: str, text: str, model_id: str,
                   similarity_boost: float, stability: float,
                   style_exaggeration: float, speed: float,
                   voice_name: str, voice_map: dict):
    voice_id = voice_map.get(voice_name, voice_name)
    # clamp parameters to valid range
    stability = max(0.0, min(stability, 1.0))
    similarity_boost = max(0.0, min(similarity_boost, 1.0))
    style_exaggeration = max(0.0, min(style_exaggeration, 1.0))
    speed = max(0.7, min(speed, 1.2))
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style_exaggeration": style_exaggeration,
            "use_speaker_boost": True,
            "speed": speed,
        },
    }
    url = f"{ELEVEN_BASE_URL}/v1/text-to-speech/{voice_id}/stream"
    r = requests.post(url, json=payload, headers=headers(api_key))
    r.raise_for_status()
    return r.content


def voice_changer_batch(api_key: str, audio_files: list, voice_name: str,
                        voice_map: dict, output_dir: str):
    voice_id = voice_map.get(voice_name, voice_name)
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for f in audio_files:
        with open(f.name, "rb") as file_data:
            files = {"audio": (os.path.basename(f.name), file_data, "audio/mpeg")}
            url = f"{ELEVEN_BASE_URL}/v1/voice-conversion/{voice_id}/stream"
            resp = requests.post(url, headers=headers(api_key), files=files)
            resp.raise_for_status()
            out_path = os.path.join(output_dir, os.path.basename(f.name))
            with open(out_path, "wb") as out_f:
                out_f.write(resp.content)
            results.append(out_path)
    return "\n".join(results)


def voice_changer_single(api_key: str, audio_bytes: bytes, voice_name: str,
                         voice_map: dict, stability: float, similarity_boost: float,
                         style_exaggeration: float) -> bytes:
    """Convert an in-memory audio to another voice and return the bytes."""
    if audio_bytes is None:
        raise gr.Error("Nessun audio da convertire: genera prima un clip TTS")

    voice_id = voice_map.get(voice_name, voice_name)
    stability = max(0.0, min(stability, 1.0))
    similarity_boost = max(0.0, min(similarity_boost, 1.0))
    style_exaggeration = max(0.0, min(style_exaggeration, 1.0))

    files = {"audio": ("input.mp3", audio_bytes, "audio/mpeg")}
    data = {
        "stability": str(stability),
        "similarity_boost": str(similarity_boost),
        "style_exaggeration": str(style_exaggeration),
    }
    url = f"{ELEVEN_BASE_URL}/v1/speech-to-speech/{voice_id}/stream"
    resp = requests.post(url, headers=headers(api_key), files=files, data=data)
    resp.raise_for_status()
    return resp.content


def clone_voice(api_key: str, name: str, description: str, audio_files: list):
    files = []
    for af in audio_files:
        files.append(("files", (os.path.basename(af.name), af, "audio/mpeg")))
    data = {"name": name, "description": description}
    url = f"{ELEVEN_BASE_URL}/v1/voices/add"
    resp = requests.post(url, headers=headers(api_key), data=data, files=files)
    resp.raise_for_status()
    res = resp.json()
    return f"Voce creata: {res.get('voice_id', '')} - {res.get('name', '')}"


def build_interface():
    with gr.Blocks(title="ElevenLabs Toolkit") as demo:
        api_key = gr.Textbox(label="API Key", type="password")
        voices_state = gr.State({})

        def refresh_voices(key):
            v = fetch_voices(key)
            return gr.update(choices=list(v.keys())), v

        with gr.Tab("TTS Avanzato"):
            refresh_btn = gr.Button("Aggiorna voci")
            voice_dropdown = gr.Dropdown(label="Voce")
            tts_text = gr.Textbox(label="Testo (SSML)", lines=5)
            model_id = gr.Dropdown([
                "eleven_monolingual_v1",
                "eleven_multilingual_v2",
            ], label="Modello vocale", value="eleven_multilingual_v2")
            similarity = gr.Slider(0, 1, 0.5, label="similarity_boost")
            stability = gr.Slider(0, 1, 0.5, label="stability")
            style = gr.Slider(0, 1, 0.0, label="style_exaggeration")
            speed = gr.Slider(0.7, 1.2, 1.0, label="speed")
            tts_btn = gr.Button("Genera Audio")
            audio_output = gr.Audio(label="Output")
            tts_state = gr.State(value=None)

            gr.Markdown("### Cambia voce dell'audio generato")
            target_voice = gr.Dropdown(label="Voce target")
            stab_conv = gr.Slider(0, 1, 0.5, label="stability")
            sim_conv = gr.Slider(0, 1, 0.5, label="similarity_boost")
            style_conv = gr.Slider(0, 1, 0.0, label="style_exaggeration")
            change_btn = gr.Button("Cambia Voce")
            changed_audio = gr.Audio(label="Output convertito")

            change_btn.click(
                lambda key, data, vname, vmap, stab, sim, sty: voice_changer_single(
                    key, data, vname, vmap, stab, sim, sty
                ),
                inputs=[api_key, tts_state, target_voice, voices_state, stab_conv, sim_conv, style_conv],
                outputs=changed_audio,
            )

            refresh_btn.click(refresh_voices, inputs=api_key,
                              outputs=[voice_dropdown, voices_state])
            refresh_btn.click(refresh_voices, inputs=api_key,
                              outputs=[target_voice, voices_state], queue=False)

            tts_btn.click(
                lambda *args: (audio := text_to_speech(*args), audio),
                inputs=[api_key, tts_text, model_id, similarity, stability, style, speed, voice_dropdown, voices_state],
                outputs=[audio_output, tts_state],
            )

        with gr.Tab("Voice Changer Batch"):
            refresh_btn2 = gr.Button("Aggiorna voci")
            voice_dropdown2 = gr.Dropdown(label="Voce")
            files_in = gr.Files(label="Audio da convertire", file_count="multiple")
            out_dir = gr.Textbox(label="Cartella output", value="converted")
            conv_btn = gr.Button("Converti")
            result_box = gr.Textbox(label="File salvati")

            refresh_btn2.click(refresh_voices, inputs=api_key,
                               outputs=[voice_dropdown2, voices_state])

            conv_btn.click(
                lambda key, files, vname, vmap, odir: voice_changer_batch(
                    key, files, vname, vmap, odir
                ),
                inputs=[api_key, files_in, voice_dropdown2, voices_state, out_dir],
                outputs=result_box,
            )

        with gr.Tab("Voice Cloning"):
            clone_name = gr.Textbox(label="Nome voce")
            clone_desc = gr.Textbox(label="Descrizione")
            clone_files = gr.Files(label="Audio", file_count="multiple")
            clone_btn = gr.Button("Crea Voce")
            clone_res = gr.Textbox(label="Risultato")

            clone_btn.click(
                lambda key, name, desc, files: clone_voice(
                    key, name, desc, files
                ),
                inputs=[api_key, clone_name, clone_desc, clone_files],
                outputs=clone_res,
            )

    return demo


def main():
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()