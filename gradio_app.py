import os
import json
import re
import requests
import gradio as gr

ELEVEN_BASE_URL = "https://api.elevenlabs.io"
PRESET_FILE = "presets.json"

def headers(api_key: str):
    return {"xi-api-key": api_key.strip()}


def load_presets() -> dict:
    if os.path.exists(PRESET_FILE):
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_presets(presets: dict):
    with open(PRESET_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)


def strip_inner_text(ssml: str) -> str:
    match = re.search(r"(<[^>]+>)(.*)(</[^>]+>)", ssml, re.DOTALL)
    if match:
        return match.group(1) + "{text}" + match.group(3)
    return ssml


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
    # clamp parameters to ElevenLabs limits
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
        presets_state = gr.State(load_presets())

        def refresh_voices(key):
            v = fetch_voices(key)
            return gr.update(choices=list(v.keys())), v

        def refresh_presets():
            p = load_presets()
            return gr.update(choices=list(p.keys())), p

        with gr.Tab("TTS Avanzato"):
            refresh_btn = gr.Button("Aggiorna voci")
            with gr.Row():
                voice_dropdown = gr.Dropdown(label="Voce")
                preset_dropdown = gr.Dropdown(label="Preset")
            tts_text = gr.Textbox(label="Testo (SSML)", lines=5)
            model_id = gr.Dropdown([
                "eleven_monolingual_v1",
                "eleven_multilingual_v2",
                "eleven_turbo_v2_5",
                "eleven_flash_v2_5"
            ], label="Modello vocale", value="eleven_multilingual_v2")
            similarity = gr.Slider(0, 1, 0.5, label="similarity_boost")
            stability = gr.Slider(0, 1, 0.5, label="stability")
            style = gr.Slider(0, 1, 0.0, label="style_exaggeration")
            speed = gr.Slider(0.7, 1.2, 1.0, label="speed")
            with gr.Row():
                tts_btn = gr.Button("Genera Audio")
                save_btn = gr.Button("Salva Preset")
            preset_name = gr.Textbox(label="Nome preset", visible=False)
            confirm_save = gr.Button("Conferma", visible=False)
            audio_output = gr.Audio(label="Output")

            refresh_btn.click(refresh_voices, inputs=api_key,
                              outputs=[voice_dropdown, voices_state])
            demo.load(refresh_presets, outputs=[preset_dropdown, presets_state])

            def apply_preset(name, presets):
                p = presets.get(name)
                if not p:
                    return [gr.update(), gr.update(), gr.update(), gr.update(), gr.update()]
                return [
                    gr.update(value=p["ssml_template"].replace("{text}", "")),
                    gr.update(value=p["similarity_boost"]),
                    gr.update(value=p["stability"]),
                    gr.update(value=p["style_exaggeration"]),
                    gr.update(value=p["speed"]),
                ]

            preset_dropdown.change(
                apply_preset,
                inputs=[preset_dropdown, presets_state],
                outputs=[tts_text, similarity, stability, style, speed],
            )

            def open_save():
                return gr.update(visible=True), gr.update(visible=True)

            save_btn.click(open_save, outputs=[preset_name, confirm_save])

            def do_save(name, text, sim, stab, sty, sp, presets):
                if not name:
                    return gr.update(), gr.update(), presets
                tpl = strip_inner_text(text)
                presets[name] = {
                    "ssml_template": tpl,
                    "similarity_boost": sim,
                    "stability": stab,
                    "style_exaggeration": sty,
                    "speed": sp,
                }
                save_presets(presets)
                return gr.update(value="", visible=False), gr.update(choices=list(presets.keys())), presets

            confirm_save.click(
                do_save,
                inputs=[preset_name, tts_text, similarity, stability, style, speed, presets_state],
                outputs=[preset_name, preset_dropdown, presets_state],
            )

            tts_btn.click(
                lambda key, txt, model, sim, stab, sty, sp, vname, vmap: text_to_speech(
                    key, txt, model, sim, stab, sty, sp, vname, vmap
                ),
                inputs=[api_key, tts_text, model_id, similarity, stability, style, speed, voice_dropdown, voices_state],
                outputs=audio_output,
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

        with gr.Tab("Presets"):
            preset_table = gr.Dataframe(headers=["name", "template", "similarity", "stability", "style", "speed"], interactive=False)

            def list_presets(presets):
                rows = [
                    [n, p["ssml_template"], p["similarity_boost"], p["stability"], p["style_exaggeration"], p["speed"]]
                    for n, p in presets.items()
                ]
                return rows

            demo.load(lambda p: list_presets(p), inputs=presets_state, outputs=preset_table)

    return demo


def main():
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()