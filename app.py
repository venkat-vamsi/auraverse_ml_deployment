# import gradio as gr
# import joblib
# import numpy as np

# # Load models
# model_stage1 = joblib.load("model_stage1.pkl")
# scaler_stage1 = joblib.load("scaler_stage1.pkl")

# model_stage2 = joblib.load("model_stage2.pkl")
# scaler_stage2 = joblib.load("scaler_stage2.pkl")


# def predict(hr, sr, temp, lux, sound, humidity):
#     # Stage 1
#     phys = np.array([[hr, sr, temp]])
#     phys_scaled = scaler_stage1.transform(phys)
#     pred = model_stage1.predict(phys_scaled)

#     is_panic = (pred[0] == -1)

#     if not is_panic:
#         return "✅ No Panic Detected", ""

#     # Stage 2
#     env = np.array([[lux, sound, humidity]])
#     env_scaled = scaler_stage2.transform(env)
#     cluster = model_stage2.predict(env_scaled)[0]

#     center = model_stage2.cluster_centers_[cluster]
#     lux_c, sound_c, hum_c = center

#     if lux_c > sound_c and lux_c > hum_c:
#         cause = "Light-triggered"
#     elif sound_c > lux_c and sound_c > hum_c:
#         cause = "Sound-triggered"
#     else:
#         cause = "Environmental-triggered"

#     return "🚨 Panic Detected!", f"Cause: {cause}"


# interface = gr.Interface(
#     fn=predict,
#     inputs=[
#         gr.Slider(50, 150, label="Heart Rate"),
#         gr.Slider(10, 100, label="Skin Resistance"),
#         gr.Slider(20, 40, label="Temperature"),
#         gr.Slider(0, 20000, label="Lux"),
#         gr.Slider(0, 120, label="Sound"),
#         gr.Slider(0, 100, label="Humidity"),
#     ],
#     outputs=["text", "text"],
#     title="AuraVerse Panic Detection System",
#     description="AI-powered panic detection + cause identification"
# )

# interface.launch(server_name="0.0.0.0", server_port=7860)

import os
import gradio as gr
import joblib
import numpy as np
from twilio.rest import Client


# --- Twilio Configuration ---
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER')
PARENT_NUMBERS = os.getenv('PARENT_NUMBERS', '').split(',')

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def make_panic_calls(cause):
    # TwiML using Amazon Polly voice
    # 'Joanna' is a popular high-quality Polly voice
    twiml_content = f"""
    <Response>
        <Say voice="Polly.Joanna">
            Emergency alert. Your child is experiencing a {cause} panic attack. 
            Please check the AuraVerse dashboard for immediate details.
        </Say>
    </Response>
    """
    
    for number in PARENT_NUMBERS:
        try:
            client.calls.create(
                to=number,
                from_=TWILIO_NUMBER,
                twiml=twiml_content
            )
        except Exception as e:
            print(f"Failed to call {number}: {e}")

# --- Existing ML Logic (Unchanged) ---
model_stage1 = joblib.load("model_stage1.pkl")
scaler_stage1 = joblib.load("scaler_stage1.pkl")
model_stage2 = joblib.load("model_stage2.pkl")
scaler_stage2 = joblib.load("scaler_stage2.pkl")

def predict(hr, sr, temp, lux, sound, humidity):
    phys = np.array([[hr, sr, temp]])
    phys_scaled = scaler_stage1.transform(phys)
    pred = model_stage1.predict(phys_scaled)

    is_panic = (pred[0] == -1)

    if not is_panic:
        return "✅ No Panic Detected", ""

    env = np.array([[lux, sound, humidity]])
    env_scaled = scaler_stage2.transform(env)
    cluster = model_stage2.predict(env_scaled)[0]
    center = model_stage2.cluster_centers_[cluster]
    lux_c, sound_c, hum_c = center

    if lux_c > sound_c and lux_c > hum_c:
        cause = "Light-triggered"
    elif sound_c > lux_c and sound_c > hum_c:
        cause = "Sound-triggered"
    else:
        cause = "Environmental-triggered"

    # --- Trigger Twilio Alert ---
    make_panic_calls(cause)

    return "🚨 Panic Detected!", f"Cause: {cause}"

# --- Gradio Interface (Unchanged) ---
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(50, 150, label="Heart Rate"),
        gr.Slider(10, 100, label="Skin Resistance"),
        gr.Slider(20, 40, label="Temperature"),
        gr.Slider(0, 20000, label="Lux"),
        gr.Slider(0, 120, label="Sound"),
        gr.Slider(0, 100, label="Humidity"),
    ],
    outputs=["text", "text"],
    title="AuraVerse Panic Detection System",
    description="AI-powered panic detection + cause identification"
)

interface.launch(server_name="0.0.0.0", server_port=7860)