import os
import gradio as gr
import tempfile
import torch
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------
# Device
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# Language Maps (NLLB codes + gTTS codes)
# -------------------
LANG_MAP = {
    "1": ("eng_Latn", "en", "English"),
    "2": ("hin_Deva", "hi", "Hindi"),
    "3": ("ben_Beng", "bn", "Bengali"),
    "4": ("tam_Taml", "ta", "Tamil"),
    "5": ("tel_Telu", "te", "Telugu"),
    "6": ("mar_Deva", "mr", "Marathi"),
    "7": ("guj_Gujr", "gu", "Gujarati"),
    "8": ("pan_Guru", "pa", "Punjabi"),
}

# -------------------
# Load NLLB Translation Model
# -------------------
model_id = "facebook/nllb-200-distilled-600M"
print("Loading NLLB-200 translation model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
model.eval()

# -------------------
# Translation
# -------------------
@torch.inference_mode()
def nllb_translate(text, src_lang, tgt_lang):
    if not text or not text.strip():
        return ""

    # Tokenize with source language
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Get target language ID
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    # Generate translation
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tgt_id,
        max_length=512,
        num_beams=10,
        early_stopping=True
    )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

# -------------------
# TTS
# -------------------
def tts_speak(text, tgt_choice):
    if not text:
        return None
    lang_iso = LANG_MAP.get(tgt_choice, ("eng_Latn", "en", ""))[1]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tmp_path = fp.name
        gTTS(text=text, lang=lang_iso).save(tmp_path)
        return tmp_path
    except Exception as e:
        print("gTTS error:", e)
        return None

# -------------------
# Gradio Function
# -------------------
def translate_text_ui(text, src_choice_str, tgt_choice_str):
    src_choice = src_choice_str.split(".")[0].strip()
    tgt_choice = tgt_choice_str.split(".")[0].strip()

    src_lang, src_iso, _ = LANG_MAP.get(src_choice, ("eng_Latn", "en", "English"))
    tgt_lang, tgt_iso, _ = LANG_MAP.get(tgt_choice, ("hin_Deva", "hi", "Hindi"))

    translated = nllb_translate(text, src_lang, tgt_lang)
    audio_path = tts_speak(translated, tgt_choice)
    return text, translated, audio_path

# -------------------
# Launch App
# -------------------
def launch_app():
    languages = [f"{k}. {v[2]}" for k, v in LANG_MAP.items()]

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            gr.Image(
                value="logo.png",
                elem_id="logo",
                show_label=False,
                scale=1,
            )
            gr.Markdown("## 🌐 **BhashaLink**: Multilingual Translation + TTS (NLLB-200)")

        with gr.Row():
            text_input = gr.Textbox(label="✍️ Enter text", placeholder="Type your sentence here...", lines=2)

        with gr.Row():
            src_dropdown = gr.Dropdown(languages, value="1. English", label="Source Language")
            tgt_dropdown = gr.Dropdown(languages, value="2. Hindi", label="Target Language")

        with gr.Row():
            text_output_orig = gr.Textbox(label="📝 Original Text")
            text_output_trans = gr.Textbox(label="🌍 Translated Text")

        with gr.Row():
            tts_audio = gr.Audio(label="🔊 Translated Speech", type="filepath", autoplay=False)

        translate_btn = gr.Button("🚀 Translate & Speak")
        translate_btn.click(
            fn=translate_text_ui,
            inputs=[text_input, src_dropdown, tgt_dropdown],
            outputs=[text_output_orig, text_output_trans, tts_audio]
        )

    # If running on Hugging Face Spaces → public mode
    if os.getenv("SPACE_ID"):
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        demo.launch()

if __name__ == "__main__":
    launch_app()
