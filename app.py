import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from brain import load_artifacts, predict_churn

# ✅ Load model and features
model, scaler, feature_names = load_artifacts()

# ✅ Set page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# ✅ Convert background image to base64
def get_base64_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

background_image = get_base64_bg("this image.jpg")

# ✅ Inject custom CSS
st.markdown(
    f"""
    <style>
    /* Background Image */
    [data-testid="stAppViewContainer"] {{
        background-image: url("{background_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Top Glowing Header */
    .custom-header {{
        position: absolute;
        top: 20px; /* moved down from the very top */
        left: 0;
        width: 100%;
        background: linear-gradient(to right, #001f3f, #0074D9);
        padding: 1rem 2rem;
        z-index: 9999;
        color: red; /* header text color */
        text-shadow: 2px 2px 4px black; /* makes red text stand out */
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4);
        animation: pulseGlow 4s infinite;
    }}

    @keyframes pulseGlow {{
        0% {{ box-shadow: 0 0 10px #0074D9; }}
        50% {{ box-shadow: 0 0 20px #39CCCC; }}
        100% {{ box-shadow: 0 0 10px #0074D9; }}
    }}

    /* Main Content Styling */
    .main {{
        margin-top: 120px !important; /* increased so content clears the header */
        background-color: rgba(255, 255, 255, 0.88);
        padding: 3rem;
        border-radius: 15px;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.25);
    }}

    h1, h2, h3 {{
        text-align: center;
        color: #1a1a2e;
    }}

    .stButton>button {{
        background-color: #008CBA;
        color: white;
        font-size: 16px;
        padding: 0.5em 1.5em;
        border-radius: 8px;
        margin-top: 1em;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Top Header
st.markdown("<div class='custom-header'>🚀 Customer Churn Prediction App</div>", unsafe_allow_html=True)

# ✅ Start Main Container
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("#### 📥 Enter Customer Features")

# ✅ User Input Form
with st.form("churn_form"):
    cols = st.columns(3)
    user_input = {}
    for idx, feat in enumerate(feature_names):
        user_input[feat] = cols[idx % 3].number_input(f"{feat}", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("🚀 Predict Churn")

# ✅ Output & Visualization
if submitted:
    label, prob = predict_churn(user_input, model, scaler, feature_names)

    st.markdown("### 🎯 Prediction Result")
    st.success(f"✅ Prediction: **{'Churn' if label == 1 else 'No Churn'}**")
    st.metric(label="📊 Churn Probability", value=f"{prob*100:.2f} %")

    # Pie Chart
    fig = px.pie(
        names=["Churn", "Not Churn"],
        values=[prob, 1 - prob],
        color_discrete_sequence=["#EF553B", "#00CC96"],
        title="🔄 Churn vs Not Churn Probability"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar Chart
    input_df = pd.DataFrame.from_dict(user_input, orient='index', columns=['Value']).reset_index()
    input_df.columns = ['Feature', 'Value']
    bar = px.bar(input_df, x="Feature", y="Value", title="📌 Input Feature Values", color="Feature")
    st.plotly_chart(bar, use_container_width=True)

# ✅ Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #ffffff; font-size: 14px;'>"
    "Made with ❤️ by <b>Ayushman Kar</b> • &copy; 2025"
    "</div>",
    unsafe_allow_html=True
)

# ✅ End Main Container
st.markdown("</div>", unsafe_allow_html=True)
