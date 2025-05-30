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
import random # NUOVO: Per generare seed casuali

ELEVEN_BASE_URL = "https://api.elevenlabs.io"
PRESET_FILE = "presets.json"
GENERATED_AUDIO_DIR = "generated_tts_audio" 
TAKE_METADATA_FILE = "take_metadata.json" # NUOVO: File per i metadati dei take salvati

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

def load_json_file(filepath: str, default_value=None):
    if default_value is None: default_value = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                gr.Warning(f"File JSON '{filepath}' corrotto o vuoto. Verrà usato il valore di default.")
                return default_value
    return default_value

def save_json_file(filepath: str, data: dict):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_presets() -> dict:
    return load_json_file(PRESET_FILE, {})

def save_presets(presets: dict):
    save_json_file(PRESET_FILE, presets)

def load_take_metadata() -> list: # I metadati sono una lista di dizionari
    return load_json_file(TAKE_METADATA_FILE, [])

def save_take_metadata(metadata_list: list):
    save_json_file(TAKE_METADATA_FILE, metadata_list)

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
    if rate == "default" and pitch == "default": return f"<speak>{text}</speak>"
    prosody_attributes = []
    if rate != "default": prosody_attributes.append(f'rate="{rate}"')
    if pitch != "default": prosody_attributes.append(f'pitch="{pitch}"')
    return f'<speak><prosody {" ".join(prosody_attributes)}>{text}</prosody></speak>'

# MODIFICATO: Aggiunto parametro seed
def text_to_speech(api_key: str, ssml_text: str, model_id: str,
                   similarity_boost: float, stability: float, style_exaggeration: float, speed_setting: float,
                   voice_name: str, voice_map: dict, 
                   previous_text_prompt: str = None, 
                   emotional_prompt_text: str = None, 
                   seed: int = None, # NUOVO parametro seed
                   output_format_param: str = DEFAULT_PCM_OUTPUT):
    voice_id = voice_map.get(voice_name, voice_name) 
    if not voice_id: raise gr.Error("Nome/ID della voce non valido.")
    
    payload = {
        "text": ssml_text, 
        "model_id": model_id,
        "voice_settings": {
            "stability": max(0.0, min(stability, 1.0)),
            "similarity_boost": max(0.0, min(similarity_boost, 1.0)),
            "style_exaggeration": max(0.0, min(style_exaggeration, 1.0)),
            "speed": max(0.7, min(speed_setting, 1.2)), # Coerente con slider 0.7-1.2
            "use_speaker_boost": True,
        }
    }
    if previous_text_prompt and previous_text_prompt.strip():
        payload["previous_text"] = previous_text_prompt.strip()
    if emotional_prompt_text and emotional_prompt_text.strip():
        payload["next_text"] = emotional_prompt_text.strip() 
    if seed is not None: # Aggiungi il seed se fornito
        payload["seed"] = int(seed)

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
            try: 
                error_json = e.response.json()
                error_detail = error_json.get('detail', {}).get('message', str(error_json))
            except json.JSONDecodeError: 
                error_detail = e.response.text
        raise gr.Error(f"Errore API TTS: {error_detail}")
    except Exception as e: raise gr.Error(f"Errore TTS imprevisto: {e}")

# ... (voice_changer_batch e clone_voice rimangono invariate) ...
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
    os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)
    with gr.Blocks(title="ElevenLabs Toolkit", theme=gr.themes.Ocean()) as demo:
        voices_state = gr.State({})
        presets_state = gr.State(load_presets())
        global_take_counter_state = gr.State(0) 
        current_batch_metadata_state = gr.State([]) # NUOVO: Per i metadati del batch corrente
        all_saved_takes_metadata_state = gr.State(load_take_metadata()) # NUOVO: Per tutti i take salvati

        with gr.Row(elem_id="api_key_top_row",equal_height=True):
            api_key_input_global = gr.Textbox(
                label="API Key ElevenLabs", type="password", lines=1, scale=4,
                placeholder="Inserisci la tua API Key di ElevenLabs qui"
            )
            refresh_button_global = gr.Button("Carica Dati", scale=1)
        
        def refresh_voices_and_presets_action(key_val):
            v_map = fetch_voices(key_val) 
            p_map = load_presets()
            v_choices = list(v_map.keys()) if v_map else []
            p_choices = list(p_map.keys()) if p_map else []
            return gr.update(choices=v_choices), v_map, gr.update(choices=p_choices), p_map

        with gr.Tab("TTS Avanzato") as tts_tab:
            with gr.Row(equal_height=True):
                voice_dropdown_tts = gr.Dropdown(label="Voce", scale=3)
                # refresh_btn_tts_tab = gr.Button("Aggiorna", scale=1, size="sm") 
                preset_dropdown_tts = gr.Dropdown(label="Preset", scale=3)

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
            
            seed_input_tts = gr.Number(label="Seed Specifico (opzionale, intero)", precision=0, value=None) # NUOVO campo per il Seed

            tts_text_input = gr.Textbox(label="Testo da Sintetizzare (SSML supportato)", lines=5, placeholder="Inserisci qui il testo o SSML...")
            previous_text_input_tts = gr.Textbox(label="Contesto Precedente (previous_text)", lines=2, placeholder="Testo che precede quello da sintetizzare...")
            emotional_prompt_input_tts = gr.Textbox(label="Prompt Emotivo (next_text)", lines=2, placeholder="Testo che segue e guida l'intonazione...")
            
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
            
            gr.Markdown("### Anteprime Audio Generate e Opzioni Take")
            audio_previews_tts = []
            save_info_buttons_tts = [] # Lista per i bottoni "Salva Info Take"
            for i in range(MAX_AUDIO_PREVIEWS):
                with gr.Row(equal_height=True): # Ogni anteprima e bottone in una riga
                    audio_previews_tts.append(gr.Audio(label=f"Anteprima {i+1}", visible=False, show_download_button=True))
                    save_info_buttons_tts.append(gr.Button(f"Salva Info Take {i+1}", visible=False, size="sm"))
            
            gr.Markdown("### Tutti i File Generati (per Download)")
            all_generated_files_dw_tts = gr.Files(label="File generati", visible=False, file_count="multiple", interactive=False)
            
            # apply_preset_func e do_save_preset_func ora includono il seed e i prompt contestuali
            def apply_preset_func(name, presets_dict_state):
                p = presets_dict_state.get(name)
                # text, prev_text, emo_prompt, seed, model, similarity, stability, style, speed, rate, pitch
                num_outputs = 11
                if not p: return [gr.update()]*num_outputs
                return [
                    p.get("input_testuale", ""), 
                    p.get("previous_text_prompt", ""), 
                    p.get("emotional_prompt", ""), 
                    p.get("seed_value"), # Può essere None se non salvato
                    p.get("model_id", "eleven_multilingual_v2"),
                    p.get("similarity_boost", 0.75), p.get("stability", 0.75),
                    p.get("style_exaggeration", 0.0), p.get("speed", 1.0), 
                    p.get("rate", "default"), p.get("pitch", "default")
                ]
            preset_dropdown_tts.change(apply_preset_func, inputs=[preset_dropdown_tts, presets_state],
                                   outputs=[tts_text_input, previous_text_input_tts, emotional_prompt_input_tts, seed_input_tts,
                                            model_id_tts, similarity_tts, stability_tts, style_tts, 
                                            speed_slider_tts, prosody_rate_tts, prosody_pitch_tts])

            def open_save_preset_ui(): return gr.update(visible=True), gr.update(visible=True)
            save_preset_btn_tts.click(open_save_preset_ui, outputs=[preset_name_input_tts, confirm_save_preset_btn_tts])

            def do_save_preset_func(name, txt, prev_text, emo_prompt, seed_val_ui, # Aggiunto seed_val_ui
                                    mid, sim, stab, sty, speed_val, rate, pitch, current_presets):
                if not name:
                    gr.Warning("Nome preset non può essere vuoto.")
                    return gr.update(visible=True), gr.update(visible=True), gr.update(), current_presets
                current_presets[name] = {
                    "input_testuale": txt, 
                    "previous_text_prompt": prev_text, 
                    "emotional_prompt": emo_prompt, 
                    "seed_value": int(seed_val_ui) if seed_val_ui is not None else None, # Salva il seed
                    "model_id": mid, "similarity_boost": sim, "stability": stab,
                    "style_exaggeration": sty, "speed": speed_val, "rate": rate, "pitch": pitch,
                }
                save_presets(current_presets)
                gr.Info(f"Preset '{name}' salvato!")
                return gr.update(value="", visible=False), gr.update(visible=False), gr.update(choices=list(current_presets.keys())), current_presets
            
            confirm_save_preset_btn_tts.click(do_save_preset_func,
                                   inputs=[preset_name_input_tts, tts_text_input, previous_text_input_tts, emotional_prompt_input_tts, 
                                           seed_input_tts, # Aggiunto input seed
                                           model_id_tts, similarity_tts, stability_tts, style_tts, 
                                           speed_slider_tts, prosody_rate_tts, prosody_pitch_tts, presets_state],
                                   outputs=[preset_name_input_tts, confirm_save_preset_btn_tts, preset_dropdown_tts, presets_state])
            
            # MODIFICATO: handle_tts_generation per gestire seed e metadati
            def handle_tts_generation(curr_api_key, txt_in, prev_text_val, emo_prompt_val, seed_input_ui_val, # NUOVO input seed_input_ui_val
                                      p_rate, p_pitch, n_gens, curr_model, 
                                      sim_val, stab_val, style_val, speed_val, v_name, v_map,
                                      current_global_take): 
                if not curr_api_key: raise gr.Error("API Key mancante!")
                if not txt_in.strip() and not prev_text_val.strip() and not emo_prompt_val.strip(): # Almeno un input testuale deve esserci
                     raise gr.Error("Almeno uno tra Testo, Contesto Precedente o Prompt Emotivo deve essere fornito.")
                if not v_name: raise gr.Error("Seleziona una voce!")
                
                ssml = construct_ssml_from_text_and_prosody(txt_in, p_rate, p_pitch) # txt_in è il testo principale per SSML
                
                batch_metadata_list = [] # Lista per i metadati di questo batch
                generated_final_paths = []
                current_date_str = datetime.datetime.now().strftime("%Y%m%d")
                sanitized_voice_name = sanitize_filename(v_name if v_name else "UnknownVoice")
                base_filename_prefix = f"{current_date_str}_{sanitized_voice_name}_TTS"
                local_take_counter = current_global_take 

                for i in range(int(n_gens)):
                    gr.Info(f"Generazione audio {i+1} di {int(n_gens)} (Take Globale {local_take_counter + 1})...")
                    temp_wav_path = None
                    
                    current_seed_for_take = None
                    if seed_input_ui_val is not None:
                        current_seed_for_take = int(seed_input_ui_val) + i
                    else:
                        current_seed_for_take = random.randint(0, 2**32 - 1)
                    
                    try:
                        local_take_counter += 1 
                        current_take_for_filename_num = local_take_counter
                        
                        temp_wav_path = text_to_speech(curr_api_key, ssml, curr_model, sim_val, stab_val, style_val, speed_val, 
                                                       v_name, v_map, 
                                                       previous_text_prompt=prev_text_val, 
                                                       emotional_prompt_text=emo_prompt_val,
                                                       seed=current_seed_for_take) # Passa il seed
                        
                        final_filename = f"{base_filename_prefix}_take{current_take_for_filename_num}.wav"
                        final_path = os.path.join(GENERATED_AUDIO_DIR, final_filename)
                        shutil.copy(temp_wav_path, final_path)
                        generated_final_paths.append(final_path)

                        # Salva metadati per questo take
                        take_meta = {
                            "filename": os.path.basename(final_path),
                            "timestamp": datetime.datetime.now().isoformat(),
                            "seed_used": current_seed_for_take,
                            "text_input": txt_in,
                            "previous_text_prompt": prev_text_val,
                            "emotional_prompt_next_text": emo_prompt_val,
                            "voice_name": v_name,
                            "model_id": curr_model,
                            "similarity_boost": sim_val,
                            "stability": stab_val,
                            "style_exaggeration": style_val,
                            "speed_setting": speed_val,
                            "prosody_rate": p_rate,
                            "prosody_pitch": p_pitch,
                            "global_take_num": current_take_for_filename_num
                        }
                        batch_metadata_list.append(take_meta)

                    except Exception as e: 
                        gr.Error(f"Errore generazione take {current_take_for_filename_num}: {e}") 
                        break 
                    finally:
                        if temp_wav_path and os.path.exists(temp_wav_path):
                            try: os.remove(temp_wav_path)
                            except Exception as e_rem: gr.Warning(f"Impossibile rimuovere file temporaneo {temp_wav_path}: {e_rem}")
                
                gr.Info(f"Generati {len(generated_final_paths)} file(s) in '{GENERATED_AUDIO_DIR}'.")
                updated_global_take_counter = local_take_counter if generated_final_paths else current_global_take
                
                preview_updates = []
                save_btn_visibility_updates = []
                for i in range(MAX_AUDIO_PREVIEWS):
                    if i < len(generated_final_paths):
                        preview_updates.append(gr.update(value=generated_final_paths[i], visible=True, label=f"Anteprima {i+1}"))
                        save_btn_visibility_updates.append(gr.update(visible=True))
                    else:
                        preview_updates.append(gr.update(value=None, visible=False, label=f"Anteprima {i+1}"))
                        save_btn_visibility_updates.append(gr.update(visible=False))
                        
                files_dw_update = gr.update(value=generated_final_paths if generated_final_paths else None, visible=bool(generated_final_paths))
                
                # Restituisce [audio_outputs] + [save_button_visibility_updates] + files_dw + global_take_counter + batch_metadata
                output_tuple = tuple(preview_updates + save_btn_visibility_updates + [files_dw_update, updated_global_take_counter, batch_metadata_list])
                return output_tuple

            tts_btn_inputs = [
                api_key_input_global, tts_text_input, previous_text_input_tts, emotional_prompt_input_tts, 
                seed_input_tts, # Aggiunto input seed
                prosody_rate_tts, prosody_pitch_tts, num_generations_tts, 
                model_id_tts, similarity_tts, stability_tts, style_tts, speed_slider_tts, 
                voice_dropdown_tts, voices_state, global_take_counter_state 
            ]
            # L'ordine degli output deve corrispondere: MAX_AUDIO_PREVIEWS per audio, MAX_AUDIO_PREVIEWS per bottoni, 1 per files, 1 per counter, 1 per metadata
            tts_btn_outputs = audio_previews_tts + save_info_buttons_tts + [all_generated_files_dw_tts, global_take_counter_state, current_batch_metadata_state] 

            generate_audio_btn_tts.click(handle_tts_generation, inputs=tts_btn_inputs, outputs=tts_btn_outputs)

            # Handler per i bottoni "Salva Info Take"
            def save_specific_take_info_action(take_idx_to_save, current_batch_meta, all_saved_meta_list):
                if not current_batch_meta or take_idx_to_save >= len(current_batch_meta):
                    gr.Warning("Metadati del take non trovati o indice non valido.")
                    return all_saved_meta_list # Non modificare la lista principale
                
                metadata_to_add = current_batch_meta[take_idx_to_save]
                
                # Evita duplicati esatti se lo stesso file/info viene salvato più volte
                # Questo controllo potrebbe essere più sofisticato se necessario
                already_exists = any(item.get("filename") == metadata_to_add.get("filename") and 
                                     item.get("seed_used") == metadata_to_add.get("seed_used") 
                                     for item in all_saved_meta_list)
                if not already_exists:
                    all_saved_meta_list.append(metadata_to_add)
                    save_take_metadata(all_saved_meta_list) # Salva l'intera lista aggiornata su file
                    gr.Info(f"Info per '{metadata_to_add.get('filename')}' salvate in {TAKE_METADATA_FILE}!")
                else:
                    gr.Info(f"Info per '{metadata_to_add.get('filename')}' già presenti.")
                return all_saved_meta_list

            for idx, btn_save_info in enumerate(save_info_buttons_tts):
                btn_save_info.click(
                    save_specific_take_info_action, 
                    inputs=[gr.State(idx), current_batch_metadata_state, all_saved_takes_metadata_state], 
                    outputs=[all_saved_takes_metadata_state] # Aggiorna lo stato con la lista completa
                )

        # ... (Tab Voice Changer Batch, Voice Cloning, Gestione Presets come prima, ma Gestione Presets ora include il seed)
        with gr.Tab("Voice Changer Batch") as vc_tab:
            refresh_btn_vc_tab = gr.Button("Aggiorna voci")
            voice_dropdown_vc = gr.Dropdown(label="Voce")
            files_in_vc = gr.Files(label="Audio da convertire (.mp3, .wav, etc.)", file_count="multiple", type="filepath")
            out_dir_vc = gr.Textbox(label="Cartella output", value="converted_audio_batch", lines=1)
            conv_btn_vc = gr.Button("Converti", variant="primary")
            result_box_vc = gr.Textbox(label="Risultati Conversione", lines=5, interactive=False)

            refresh_btn_vc_tab.click(refresh_voices_and_presets_action, inputs=api_key_input_global,
                               outputs=[voice_dropdown_vc, voices_state, preset_dropdown_tts, presets_state])

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
                headers=["Nome", "Testo", "Contesto Prec.", "Prompt Emotivo", "Seed", "Modello ID", "Similarity", "Stability", "Style", "Speed", "Rate", "Pitch"], 
                interactive=False 
            )
            with gr.Row():
                delete_preset_dropdown_manage = gr.Dropdown(label="Seleziona Preset da Eliminare", scale=3)
                confirm_delete_btn_manage = gr.Button("Elimina Preset Selezionato", scale=1, variant="stop")

            def list_presets_for_ui(presets_data): # Ora include il seed
                rows, names = [], list(presets_data.keys())
                for name in names:
                    p = presets_data[name]
                    rows.append([name, p.get("input_testuale",""), 
                                 p.get("previous_text_prompt",""), p.get("emotional_prompt",""),
                                 p.get("seed_value"), # Visualizza il seed
                                 p.get("model_id",""), p.get("similarity_boost",""), p.get("stability",""), 
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
            ).then(
                lambda p_state_upd: gr.update(choices=list(p_state_upd.keys())), inputs=presets_state, outputs=preset_dropdown_tts
            )
        
        refresh_button_global.click(
            refresh_voices_and_presets_action,
            inputs=api_key_input_global,
            outputs=[voice_dropdown_tts, voices_state, preset_dropdown_tts, presets_state] 
        )
        demo.load(refresh_voices_and_presets_action, inputs=api_key_input_global, outputs=[voice_dropdown_tts, voices_state, preset_dropdown_tts, presets_state])

    return demo

def main():
    demo = build_interface()
    demo.launch(debug=True, share=True)

if __name__ == "__main__":
   main()