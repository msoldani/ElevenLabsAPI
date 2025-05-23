import os
import json
import requests
import gradio as gr
import tempfile
import wave
import traceback 
import datetime 
import shutil 
import re 

ELEVEN_BASE_URL = "https://api.elevenlabs.io"
PRESET_FILE = "presets.json"
GENERATED_AUDIO_DIR = "generated_tts_audio" 

PCM_FORMATS = {
    "pcm_16000": 16000, "pcm_22050": 22050,
    "pcm_24000": 24000, "pcm_44100": 44100,
}
DEFAULT_PCM_OUTPUT = "pcm_44100"
MAX_AUDIO_PREVIEWS = 3 

def headers(api_key: str):
    return {"xi-api-key": api_key.strip()}

def sanitize_filename(name: str, default_if_empty="audio", max_length=50) -> str:
    if not name: name = default_if_empty
    name = str(name) 
    name = re.sub(r'[^\w\s\-\.]', '', name).strip()
    name = re.sub(r'[-\s]+', '_', name)
    name = re.sub(r'_+', '_', name) 
    return name[:max_length].strip('_')

def load_presets() -> dict:
    if os.path.exists(PRESET_FILE):
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                gr.Warning(f"File presets '{PRESET_FILE}' corrotto o vuoto.")
                return {}
    return {}

def save_presets(presets: dict):
    with open(PRESET_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)

def save_pcm_to_wav(pcm_data, framerate, channels=1, sampwidth=2):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(channels); wf.setsampwidth(sampwidth); 
            wf.setframerate(framerate); wf.writeframes(pcm_data)
        return temp_file.name
    except Exception as e:
        if os.path.exists(temp_file.name):
            try: os.remove(temp_file.name)
            except OSError: pass 
        raise IOError(f"Errore salvataggio WAV: {e}")

def fetch_voices(api_key: str) -> dict:
    if not api_key: return {}
    try:
        resp = requests.get(f"{ELEVEN_BASE_URL}/v1/voices", headers=headers(api_key))
        resp.raise_for_status()
        data = resp.json()
        return {v["name"]: v["voice_id"] for v in data.get("voices", [])}
    except requests.exceptions.RequestException as e:
        gr.Warning(f"Impossibile caricare le voci: {e}")
        return {}
    except Exception as e:
        gr.Warning(f"Errore caricamento voci: {e}")
        return {}

def construct_ssml_from_text_and_prosody(text: str, rate: str, pitch: str) -> str:
    escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if rate == "default" and pitch == "default": return f"<speak>{escaped_text}</speak>"
    prosody_attributes = []
    if rate != "default": prosody_attributes.append(f'rate="{rate}"')
    if pitch != "default": prosody_attributes.append(f'pitch="{pitch}"')
    return f'<speak><prosody {" ".join(prosody_attributes)}>{escaped_text}</prosody></speak>'

def text_to_speech(api_key: str, ssml_text: str, model_id: str,
                   similarity_boost: float, stability: float, style_exaggeration: float, speed_setting: float,
                   voice_name: str, voice_map: dict, output_format_param: str = DEFAULT_PCM_OUTPUT):
    voice_id = voice_map.get(voice_name, voice_name) 
    if not voice_id: raise gr.Error("Nome/ID della voce non valido.")
    payload = {
        "text": ssml_text, "model_id": model_id,
        "voice_settings": {
            "stability": max(0.0, min(stability, 1.0)),
            "similarity_boost": max(0.0, min(similarity_boost, 1.0)),
            "style_exaggeration": max(0.0, min(style_exaggeration, 1.0)),
            "speed": max(0.7, min(speed_setting, 1.2)), 
            "use_speaker_boost": True,
        },
    }
    url = f"{ELEVEN_BASE_URL}/v1/text-to-speech/{voice_id}/stream?output_format={output_format_param}"
    try:
        r = requests.post(url, json=payload, headers=headers(api_key), stream=True)
        r.raise_for_status()
        if output_format_param in PCM_FORMATS:
            framerate = PCM_FORMATS[output_format_param]
            audio_data = b''.join(r.iter_content(chunk_size=1024*1024))
            return save_pcm_to_wav(audio_data, framerate=framerate)
        else: 
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") 
            with open(temp_file.name, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
            return temp_file.name
    except requests.exceptions.RequestException as e:
        error_detail = str(e)
        if e.response is not None:
            try: error_detail = e.response.json().get('detail', {}).get('message', e.response.text)
            except json.JSONDecodeError: error_detail = e.response.text
        raise gr.Error(f"Errore API TTS: {error_detail}")
    except Exception as e: raise gr.Error(f"Errore TTS imprevisto: {e}")

def voice_changer_batch(api_key: str, audio_files: list, voice_name: str, voice_map: dict, output_dir: str):
    voice_id = voice_map.get(voice_name, voice_name)
    if not voice_id: raise gr.Error("Voce non valida per Voice Changer.")
    os.makedirs(output_dir, exist_ok=True)
    results = []
    if not audio_files: return "Nessun file audio fornito."
    for f_obj in audio_files:
        file_path = f_obj.name
        try:
            with open(file_path, "rb") as file_data:
                files_payload = {"audio": (os.path.basename(file_path), file_data, "audio/mpeg")}
                url = f"{ELEVEN_BASE_URL}/v1/voice-conversion/{voice_id}/stream"
                resp = requests.post(url, headers=headers(api_key), files=files_payload)
                resp.raise_for_status()
                out_filename = f"vc_{sanitize_filename(voice_name)}_{os.path.basename(file_path)}"
                out_path = os.path.join(output_dir, out_filename)
                with open(out_path, "wb") as out_f: out_f.write(resp.content)
                results.append(out_path)
        except Exception as e:
            gr.Warning(f"Errore conversione di {os.path.basename(file_path)}: {e}")
            results.append(f"FALLITO: {os.path.basename(file_path)}")
    return "\n".join(results) if results else "Nessun file elaborato."

def clone_voice(api_key: str, name: str, description: str, audio_files_list: list):
    opened_files_to_close = []
    try:
        files_payload = []
        for af_obj in audio_files_list:
            f = open(af_obj.name, "rb")
            opened_files_to_close.append(f)
            files_payload.append(("files", (os.path.basename(af_obj.name), f, "audio/mpeg")))
        data = {"name": name, "description": description}
        url = f"{ELEVEN_BASE_URL}/v1/voices/add"
        resp = requests.post(url, headers=headers(api_key), data=data, files=files_payload)
        resp.raise_for_status()
        res = resp.json()
        return f"Voce creata: {res.get('voice_id', '')} - {res.get('name', '')}"
    except requests.exceptions.RequestException as e:
        error_detail = str(e)
        if e.response is not None:
            try: error_detail = e.response.json().get('detail', {}).get('message', e.response.text)
            except json.JSONDecodeError: error_detail = e.response.text
        raise gr.Error(f"API Error cloning: {error_detail}")
    except Exception as e: raise gr.Error(f"Unexpected error cloning: {e}")
    finally:
        for f_handle in opened_files_to_close:
            try: f_handle.close()
            except: pass

def build_interface():
    # Operazioni di setup iniziali
    os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)

    with gr.Blocks(title="ElevenLabs Toolkit", theme=gr.themes.Ocean()) as demo:
        # Stati globali (non visibili, definiti presto)
        voices_state = gr.State({})
        presets_state = gr.State(load_presets())

        # MODIFICATO: Riga superiore per API Key e pulsante di aggiornamento globale
        with gr.Row(elem_id="api_key_top_row",equal_height=True):
            api_key_input_global = gr.Textbox(
                label="API Key ElevenLabs", 
                type="password", 
                lines=1, 
                scale=4, # Textbox occupa più spazio
                placeholder="Inserisci la tua API Key di ElevenLabs qui"
            )
            refresh_button_global = gr.Button( # Pulsante globale per aggiornare
                "Carica Dati", # Etichetta per aggiornamento globale
                scale=1, 
                
            )
        
        # Funzione helper per aggiornare le voci e i preset (usata da entrambi i pulsanti di refresh)
        def refresh_voices_and_presets_action(key_val): # Rinominata per evitare conflitto con nome componente
            v_map = fetch_voices(key_val) 
            p_map = load_presets()
            v_choices = list(v_map.keys()) if v_map else []
            p_choices = list(p_map.keys()) if p_map else []
            # Gli output saranno mappati ai componenti specifici nel .click()
            return gr.update(choices=v_choices), v_map, gr.update(choices=p_choices), p_map

        # Le definizioni delle schede (Tabs) iniziano qui
        with gr.Tab("TTS Avanzato") as tts_tab: # Assegna un nome alla tab se devi referenziarla
            with gr.Row(equal_height=True):
                voice_dropdown_tts = gr.Dropdown(label="Voce", scale=3) # Nome univoco per questo dropdown
                #refresh_btn_tts_tab = gr.Button("Aggiorna", scale=1, size="sm") # Pulsante specifico della tab
                preset_dropdown_tts = gr.Dropdown(label="Preset", scale=3) # Nome univoco

            with gr.Row():
                prosody_rate_tts = gr.Dropdown(label="Prosody Rate", choices=["default", "x-slow", "slow", "medium", "fast", "x-fast"], value="default")
                prosody_pitch_tts = gr.Dropdown(label="Prosody Pitch", choices=["default", "x-low", "low", "medium", "high", "x-high"], value="default")
            
            gr.Markdown("#### Parametri Voce")
            with gr.Row():
                similarity_tts = gr.Slider(minimum=0, maximum=1, value=0.75, label="Similarity Boost", step=0.05)
                stability_tts = gr.Slider(minimum=0, maximum=1, value=0.75, label="Stability", step=0.05)
            with gr.Row():
                style_tts = gr.Slider(minimum=0, maximum=1, value=0.0, label="Style Exaggeration", step=0.05)
                speed_slider_tts = gr.Slider(minimum=0.7, maximum=1.2, value=1.0, label="Speed (Voice Setting)", step=0.05)
            
            tts_text_input = gr.Textbox(label="Testo da Sintetizzare", lines=5, placeholder="Inserisci qui il testo...")
            num_generations_tts = gr.Number(label="Numero di generazioni", value=1, minimum=1, maximum=MAX_AUDIO_PREVIEWS * 2, step=1, precision=0)

            model_id_tts = gr.Dropdown(
                choices=["eleven_monolingual_v1", "eleven_multilingual_v1", "eleven_multilingual_v2", 
                         "eleven_turbo_v2", "eleven_turbo_v2_5", "eleven_english_sts_v2"], 
                label="Modello vocale", value="eleven_multilingual_v2"
            )
            
            with gr.Row():
                generate_audio_btn_tts = gr.Button("Genera Audio (.wav)", variant="primary")
                save_preset_btn_tts = gr.Button("Salva Preset")
            
            preset_name_input_tts = gr.Textbox(label="Nome preset", visible=False, lines=1)
            confirm_save_preset_btn_tts = gr.Button("Conferma Salvataggio", visible=False, variant="primary")
            
            gr.Markdown("### Anteprime Audio Generate")
            with gr.Row(): 
                audio_previews_tts = []
                for i in range(MAX_AUDIO_PREVIEWS):
                    audio_previews_tts.append(gr.Audio(label=f"Anteprima {i+1}", visible=False, show_download_button=True))
            
            gr.Markdown("### Tutti i File Generati (per Download)")
            all_generated_files_dw_tts = gr.Files(label="File generati", visible=False, file_count="multiple", interactive=False)

            # Eventi per la tab TTS Avanzato
            #refresh_btn_tts_tab.click(
                #refresh_voices_and_presets_action, 
                #inputs=api_key_input_global, # Usa la API key globale
                #outputs=[voice_dropdown_tts, voices_state, preset_dropdown_tts, presets_state]
            #)
            
            def apply_preset_func(name, presets_dict_state):
                p = presets_dict_state.get(name)
                num_outputs = 8 
                if not p: return [gr.update()]*num_outputs
                return [
                    p.get("input_testuale", ""), p.get("model_id", "eleven_multilingual_v2"),
                    p.get("similarity_boost", 0.75), p.get("stability", 0.75),
                    p.get("style_exaggeration", 0.0), p.get("speed", 1.0), 
                    p.get("rate", "default"), p.get("pitch", "default"),
                ]
            preset_dropdown_tts.change(apply_preset_func, inputs=[preset_dropdown_tts, presets_state],
                                   outputs=[tts_text_input, model_id_tts, similarity_tts, stability_tts, style_tts, speed_slider_tts, prosody_rate_tts, prosody_pitch_tts])

            def open_save_preset_ui(): return gr.update(visible=True), gr.update(visible=True)
            save_preset_btn_tts.click(open_save_preset_ui, outputs=[preset_name_input_tts, confirm_save_preset_btn_tts])

            def do_save_preset_func(name, txt, mid, sim, stab, sty, speed_val, rate, pitch, current_presets):
                if not name:
                    gr.Warning("Nome preset non può essere vuoto.")
                    return gr.update(visible=True), gr.update(visible=True), gr.update(), current_presets
                current_presets[name] = {
                    "input_testuale": txt, "model_id": mid, "similarity_boost": sim, "stability": stab,
                    "style_exaggeration": sty, "speed": speed_val, "rate": rate, "pitch": pitch,
                }
                save_presets(current_presets)
                gr.Info(f"Preset '{name}' salvato!")
                return gr.update(value="", visible=False), gr.update(visible=False), gr.update(choices=list(current_presets.keys())), current_presets
            confirm_save_preset_btn_tts.click(do_save_preset_func,
                                   inputs=[preset_name_input_tts, tts_text_input, model_id_tts, similarity_tts, stability_tts, style_tts, speed_slider_tts, prosody_rate_tts, prosody_pitch_tts, presets_state],
                                   outputs=[preset_name_input_tts, confirm_save_preset_btn_tts, preset_dropdown_tts, presets_state])
            
            def handle_tts_generation(curr_api_key, txt_in, p_rate, p_pitch, n_gens, curr_model, 
                                      sim_val, stab_val, style_val, speed_val, v_name, v_map):
                if not curr_api_key: raise gr.Error("API Key mancante!")
                if not txt_in.strip(): raise gr.Error("Testo non può essere vuoto!")
                if not v_name: raise gr.Error("Seleziona una voce!")
                ssml = construct_ssml_from_text_and_prosody(txt_in, p_rate, p_pitch)
                generated_final_paths = []
                current_date_str = datetime.datetime.now().strftime("%Y%m%d")
                sanitized_voice_name = sanitize_filename(v_name if v_name else "UnknownVoice")
                base_filename_prefix = f"{current_date_str}_{sanitized_voice_name}_TTS"

                for i in range(int(n_gens)):
                    gr.Info(f"Generazione audio {i+1} di {int(n_gens)}...")
                    temp_wav_path = None
                    try:
                        temp_wav_path = text_to_speech(curr_api_key, ssml, curr_model, sim_val, stab_val, style_val, speed_val, v_name, v_map)
                        take_num = i + 1
                        final_filename = f"{base_filename_prefix}_take{take_num}.wav"
                        final_path = os.path.join(GENERATED_AUDIO_DIR, final_filename)
                        shutil.copy(temp_wav_path, final_path)
                        generated_final_paths.append(final_path)
                    except Exception as e: 
                        gr.Error(f"Errore generazione {i+1}: {e}") 
                        break 
                    finally:
                        if temp_wav_path and os.path.exists(temp_wav_path):
                            try: os.remove(temp_wav_path)
                            except Exception as e_rem: gr.Warning(f"Impossibile rimuovere file temporaneo {temp_wav_path}: {e_rem}")
                
                gr.Info(f"Generati {len(generated_final_paths)} file(s) in '{GENERATED_AUDIO_DIR}'.")
                preview_updates = []
                for i in range(MAX_AUDIO_PREVIEWS):
                    if i < len(generated_final_paths):
                        preview_updates.append(gr.update(value=generated_final_paths[i], visible=True, label=f"Anteprima {i+1}"))
                    else:
                        preview_updates.append(gr.update(value=None, visible=False, label=f"Anteprima {i+1}"))
                files_dw_update = gr.update(value=generated_final_paths if generated_final_paths else None, visible=bool(generated_final_paths))
                return tuple(preview_updates + [files_dw_update])

            tts_outputs = audio_previews_tts + [all_generated_files_dw_tts]
            generate_audio_btn_tts.click(handle_tts_generation,
                          inputs=[api_key_input_global, tts_text_input, prosody_rate_tts, prosody_pitch_tts, num_generations_tts, 
                                  model_id_tts, similarity_tts, stability_tts, style_tts, speed_slider_tts, 
                                  voice_dropdown_tts, voices_state],
                          outputs=tts_outputs)

        with gr.Tab("Voice Changer Batch") as vc_tab:
            refresh_btn_vc_tab = gr.Button("Aggiorna voci")
            voice_dropdown_vc = gr.Dropdown(label="Voce")
            files_in_vc = gr.Files(label="Audio da convertire (.mp3, .wav, etc.)", file_count="multiple", type="filepath")
            out_dir_vc = gr.Textbox(label="Cartella output", value="converted_audio_batch", lines=1)
            conv_btn_vc = gr.Button("Converti", variant="primary")
            result_box_vc = gr.Textbox(label="Risultati Conversione", lines=5, interactive=False)

            refresh_btn_vc_tab.click(refresh_voices_and_presets_action, inputs=api_key_input_global, # Usa API key globale
                               outputs=[voice_dropdown_vc, voices_state, preset_dropdown_tts, presets_state]) # Aggiorna anche il preset_dropdown principale

            def handle_vc_batch(curr_api_key, files_list, vname, vmap, odir):
                if not curr_api_key: raise gr.Error("API Key mancante!")
                if not files_list: raise gr.Error("Nessun file audio fornito.")
                if not vname: raise gr.Error("Seleziona una voce.")
                return voice_changer_batch(curr_api_key, files_list, vname, vmap, odir)
            conv_btn_vc.click(handle_vc_batch, inputs=[api_key_input_global, files_in_vc, voice_dropdown_vc, voices_state, out_dir_vc], outputs=result_box_vc)

        with gr.Tab("Voice Cloning") as clone_tab:
            clone_name_clone = gr.Textbox(label="Nome voce", lines=1, placeholder="Nome per la nuova voce clonata")
            clone_desc_clone = gr.Textbox(label="Descrizione", lines=1, placeholder="Breve descrizione della voce (opzionale)")
            clone_files_clone = gr.Files(label="File Audio per Cloning (min 1, max 25)", file_count="multiple", type="filepath")
            clone_btn_clone = gr.Button("Crea Voce Clonata", variant="primary")
            clone_res_clone = gr.Textbox(label="Risultato Cloning", interactive=False, lines=1)
            
            def handle_clone(curr_api_key, name_val, desc_val, files_list_val):
                if not curr_api_key: raise gr.Error("API Key mancante!")
                if not name_val.strip(): raise gr.Error("Nome voce non può essere vuoto.")
                if not files_list_val : raise gr.Error("Fornire almeno un file audio.")
                num_f = len(files_list_val)
                if not (1 <= num_f <= 25): raise gr.Error(f"Caricare da 1 a 25 file (caricati: {num_f}).")
                return clone_voice(curr_api_key, name_val, desc_val, files_list_val)
            clone_btn_clone.click(handle_clone, inputs=[api_key_input_global, clone_name_clone, clone_desc_clone, clone_files_clone], outputs=clone_res_clone)

        with gr.Tab("Gestione Presets") as presets_manage_tab:
            presets_df_manage = gr.DataFrame(
                headers=["Nome", "Testo (Input)", "Modello ID", "Similarity", "Stability", "Style", "Speed", "Rate", "Pitch"], 
                interactive=False 
            )
            with gr.Row():
                delete_preset_dropdown_manage = gr.Dropdown(label="Seleziona Preset da Eliminare", scale=3)
                confirm_delete_btn_manage = gr.Button("Elimina Preset Selezionato", scale=1, variant="stop")

            def list_presets_for_ui(presets_data):
                rows, names = [], list(presets_data.keys())
                for name in names:
                    p = presets_data[name]
                    rows.append([name, p.get("input_testuale",""), p.get("model_id",""), 
                                 p.get("similarity_boost",""), p.get("stability",""), 
                                 p.get("style_exaggeration",""), p.get("speed",""), 
                                 p.get("rate",""), p.get("pitch","")])
                return rows, gr.update(choices=names, value=None) 

            @presets_state.change(inputs=presets_state, outputs=[presets_df_manage, delete_preset_dropdown_manage])
            def update_preset_management_views(current_presets): return list_presets_for_ui(current_presets)
            
            demo.load(lambda p_load: list_presets_for_ui(p_load), inputs=presets_state, outputs=[presets_df_manage, delete_preset_dropdown_manage])
            
            def delete_selected_p(name_del, current_ps):
                if not name_del: gr.Warning("Nessun preset selezionato."); return current_ps
                if name_del in current_ps:
                    del current_ps[name_del]
                    save_presets(current_ps)
                    gr.Info(f"Preset '{name_del}' eliminato.")
                else: gr.Warning(f"Preset '{name_del}' non trovato.")
                return current_ps
            
            confirm_delete_btn_manage.click(
                delete_selected_p, inputs=[delete_preset_dropdown_manage, presets_state], outputs=[presets_state]
            ).then( # Aggiorna il dropdown dei preset nella tab TTS Avanzato
                lambda p_state_upd: gr.update(choices=list(p_state_upd.keys())), inputs=presets_state, outputs=preset_dropdown_tts
            )
        
        # Collegamento del pulsante di aggiornamento globale (definito DOPO che i componenti di output sono stati creati)
        refresh_button_global.click(
            refresh_voices_and_presets_action,
            inputs=api_key_input_global,
            outputs=[voice_dropdown_tts, voices_state, preset_dropdown_tts, presets_state] 
            # Questo aggiorna i dropdown nella tab TTS e gli stati globali.
            # Altri dropdown (es. in Voice Changer) hanno il loro pulsante di refresh dedicato
            # o potrebbero essere aggiornati tramite .change() su voices_state/presets_state se necessario.
        )
        # Caricamento iniziale globale (già gestito dal demo.load nella tab TTS per i dropdown principali)
        # Un demo.load specifico per il pulsante globale non è necessario se quello nella tab TTS si occupa del caricamento iniziale.
        # Tuttavia, per coerenza, il demo.load globale può essere questo:
        demo.load(refresh_voices_and_presets_action, inputs=api_key_input_global, outputs=[voice_dropdown_tts, voices_state, preset_dropdown_tts, presets_state])

    return demo

def main():
    demo = build_interface()
    demo.launch(debug=True)

if __name__ == "__main__":
   main()