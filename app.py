import gradio as gr
import joblib
import numpy as np

# Load models
model_stage1 = joblib.load("model_stage1.pkl")
scaler_stage1 = joblib.load("scaler_stage1.pkl")

model_stage2 = joblib.load("model_stage2.pkl")
scaler_stage2 = joblib.load("scaler_stage2.pkl")


def predict(hr, sr, temp, lux, sound, humidity):
    # Stage 1
    phys = np.array([[hr, sr, temp]])
    phys_scaled = scaler_stage1.transform(phys)
    pred = model_stage1.predict(phys_scaled)

    is_panic = (pred[0] == -1)

    if not is_panic:
        return "✅ No Panic Detected", ""

    # Stage 2
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

    return "🚨 Panic Detected!", f"Cause: {cause}"


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