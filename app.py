import gradio as gr
import joblib
import pandas as pd

# Load model
model = joblib.load("tiktok_viral_predictor.pkl")

# Prediction function
def predict_viral(category, video_length, hashtags_count, sound_used, upload_hour,
                  user_followers, user_following, user_likes, region):

    # Automatic trend mapping
    trend_map = {
        "Dance": 0.9, "Comedy": 0.8, "Tutorial": 0.6, 
        "Fitness": 0.7, "Lifestyle": 0.5, "Gaming": 0.65, "Music": 0.85
    }
    category_trend_score = trend_map.get(category, 0.5)

    # Create input dataframe
    data = pd.DataFrame([{
        "Category": category,
        "Duration": video_length,
        "Hashtags_Count": hashtags_count,
        "Sound_Used": sound_used,
        "Upload_Hour": upload_hour,
        "Followers": user_followers,
        "Following": user_following,
        "User_Likes": user_likes,
        "Region": region,
        "Category_TrendScore": category_trend_score
    }])

    # Make prediction
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100

    # Output message
    result = "Viral!" if pred == 1 else "Not viral"
    return f"{result}\nProbability: {prob:.2f}%"

# UI Inputs
inputs = [
    gr.Dropdown(choices=["Dance", "Comedy", "Tutorial", "Fitness", "Lifestyle", "Gaming", "Music"], label="🎬 Category", value="Dance"),
    gr.Number(label="📹 Video Length (seconds)", value=30),
    gr.Number(0, 10, value=3, step=1, label="#️⃣ Hashtags Count"),
    gr.Dropdown(choices=["Trending", "Original"], label="🎵 Sound Used", value="Trending"),
    gr.Number(0, 23, value=18, step=1, label="🕒 Upload Hour"),
    gr.Number(label="👥 User Followers", value=5000),
    gr.Number(label="👣 User Following", value=300),
    gr.Number(label="❤️ Total Likes", value=1000),
    gr.Dropdown(choices=["Europe", "USA", "Asia", "South America", "Africa"], label="🌍 Region", value="Europe"),
]

# Output
outputs = gr.Textbox(label="Prediction")

# Launch app
demo = gr.Interface(
    fn=predict_viral,
    inputs=inputs,
    outputs=outputs,
    title=" TikTok Viral Predictor",
    description="Describe your video idea and see how likely it is to go viral ",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
