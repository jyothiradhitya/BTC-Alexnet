import os
import json
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import io  # Needed for handling image bytes if necessary
#from datetime import datetime

# AI Model Imports
import google.generativeai as gen_ai

# Brain Tumor Classification Imports
import torch
import torch.nn as nn
from torchvision import transforms, models


import streamlit.components.v1 as components

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="TumorScanAI.com",
    page_icon=":brain:",
    layout="centered",
)

# Load the Google API Key
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# NEW WAY: Use Streamlit Secrets
try:
    # When deployed on Streamlit Cloud, it reads from the secrets you provide
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
     # Fallback for local development (optional, if you want to keep .env for local)
     # If you choose this, uncomment load_dotenv() above
     # load_dotenv()
     # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
     # print("API Key not found in Streamlit secrets, trying .env (for local dev)") # Debug print
     # OR simply raise an error if secrets are expected
     st.error("GOOGLE_API_KEY not found in Streamlit secrets. Please configure secrets for deployment.")
     st.stop()


if not GOOGLE_API_KEY:
    st.error("Google API Key could not be loaded. Please configure secrets for deployment or check local .env.")
    st.stop() # Stop execution if key is missing



try:
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    chat_model = gen_ai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error(f"Failed to configure Google Gemini: {e}")
    st.stop()

# Set up the brain tumor classification model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'alexnet_brain_tumor_classification.pth'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Make sure it's in the same directory.")
        st.stop()


 # Define transformations (should match training)
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Model Architecture
    alexnet = models.alexnet(weights=None) # Load architecture, not pretrained weights here
    num_classes = 4 # glioma, meningioma, notumor, pituitary
    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, num_classes)

    # Load Trained Weights
    alexnet.load_state_dict(torch.load(model_path, map_location=device))
    alexnet = alexnet.to(device)
    alexnet.eval() # Set model to evaluation mode


    class_names = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary'] # Ensure order matches training

except Exception as e:
    st.error(f"Error loading the brain tumor classification model: {e}")
    st.stop()


# Function to predict the class of an uploaded image

def predict_image(image_pil, model):
    """Predicts the class of a PIL Image."""
    try:
        image_tensor = image_transforms(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1) # Get probabilities
            confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        return predicted_class, confidence.item()
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def load_users():
    """Loads user data from users.json"""
    if os.path.exists("users.json"):
        try:
            with open("users.json", "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error("Error reading users.json. File might be corrupted.")
            return {}
        except Exception as e:
            st.error(f"An error occurred loading users: {e}")
            return {}
    return {}

def save_users(users):
    """Saves user data to users.json"""
    try:
        with open("users.json", "w") as f:
            json.dump(users, f, indent=4) # Use indent for readability
    except Exception as e:
        st.error(f"An error occurred saving users: {e}")



def assess_tumor_emergency(predicted_class):
    """Provides information based on the predicted tumor class."""
    emergency_info = {
        "Pituitary": {
            "emergency_level": "Medium",
            "action": "Visit a doctor to discuss treatment options. Pituitary tumors can affect hormone levels and require medical attention.",
            "origin": "Arise from the pituitary gland, a small gland at the base of the brain responsible for hormone production.",
            "description": """
                Pituitary tumors are growths in the pituitary gland. Common symptoms include headaches, vision problems, hormonal imbalances (fatigue, weight changes, menstrual irregularities, etc.). Diagnosis involves blood tests and imaging (MRI/CT). Treatment options: Surgery (often through the nose), Radiation Therapy, Medication to control hormone levels or shrink the tumor, or Observation for small, non-symptomatic tumors. Prognosis is generally good, especially for benign tumors, with a high 5-year survival rate.
            """
        },
        "Glioma": {
            "emergency_level": "High",
            "action": "Seek immediate medical consultation and treatment options, as gliomas are often aggressive and require prompt attention.",
            "origin": "Gliomas arise from glial cells, which support and protect neurons in the brain and spinal cord.",
            "description": """
                Gliomas are common brain tumors originating from glial cells. They are graded I (least aggressive) to IV (most aggressive, like Glioblastoma). Symptoms depend on location/size: Headaches, Seizures, Nausea, Vision/Hearing changes, Weakness/Numbness, Cognitive/Personality changes. Diagnosis uses imaging (MRI/CT) and Biopsy. Treatment involves Surgery, Radiation, and Chemotherapy. Prognosis varies greatly by grade; high-grade gliomas have a more challenging prognosis and require aggressive treatment.
            """
        },
        "Meningioma": {
            "emergency_level": "Low",
            "action": "Consult a doctor for routine follow-ups and management options, as most meningiomas are benign and treatable.",
            "origin": "Meningiomas arise from the meninges, the membranes that cover the brain and spinal cord.",
            "description": """
                Meningiomas grow from the meninges (brain/spinal cord coverings). Most are benign and slow-growing. Symptoms (if present) depend on size/location: Headaches, Seizures, Vision changes, Weakness, Balance problems. Diagnosis via imaging (MRI/CT), sometimes biopsy. Treatment: Observation (for small, asymptomatic ones), Surgery (common and often curative), Radiation (for remnants, recurrence, or inoperable tumors). Prognosis is generally excellent for benign cases fully removed surgically.
            """
        },
        "Notumor": { # Changed from "No Tumor" to match class_names
            "emergency_level": "None",
            "action": "No tumor detected based on the image analysis. Maintain regular health check-ups.",
            "origin": "Analysis indicates no signs of the tumor types the model is trained on.",
            "description": """
                The analysis did not detect features characteristic of Glioma, Meningioma, or Pituitary tumors in the provided image. This suggests the absence of these specific conditions. It's important to remember this AI analysis is not a substitute for a professional medical diagnosis. Continue with regular health screenings and consult a doctor if you have any symptoms or concerns.
            """
        }
    }
    # Use .get for safer dictionary access with a default fallback
    return emergency_info.get(predicted_class, {
        "emergency_level": "Unknown",
        "action": "Prediction class not recognized. Consult a healthcare professional.",
        "origin": "N/A",
        "description": "The model returned a class not defined in the emergency information."
    })


def translate_role_for_streamlit(user_role):
    """Maps Gemini roles to Streamlit roles."""
    if user_role == "model":
        return "assistant"
    else:
        return "user"


# --- Page Navigation ---

if "page" not in st.session_state:
    st.session_state.page = "main"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun() 


# --- Page Definitions ---

def main_page():
    st.markdown("""
        <style>
        /* Basic Styling */
        .stApp { background-color: white; color: black; }
        .stButton>button {
            width: 100%; height: 70px; font-size: 20px;
            border: 2px solid black; background-color: white; color: black;
            margin-bottom: 10px; /* Add space between buttons */
        }
        .stButton>button:hover { background-color: #f0f0f0; color: black; border-color: #555; }
        h1, h3 { text-align: center; color: black; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>MEDICAL AI PLATFORM</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Advanced brain tumor classification and medical chatbot assistance powered by artificial intelligence.</h3>", unsafe_allow_html=True)
    st.markdown("---") # Add a separator

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign In"):
            navigate_to("signin")
    with col2:
        if st.button("Register"):
            navigate_to("register")


def register_page():
    st.title("Register")
    users = load_users()

    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if not username or not password or not confirm_password:
                st.error("All fields are required.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif username in users:
                st.error("Username already exists. Please choose another.")
            else:
                users[username] = password # In real app, hash the password!
                save_users(users)
                st.success("Registration successful! Please Sign In.")
                navigate_to("signin")

    if st.button("Back to Sign In"):
        navigate_to("signin")


def signin_page():
    st.title("Sign In")
    users = load_users()

    with st.form("signin_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")

        if submitted:
            if not username or not password:
                st.error("Both username and password are required.")
            # In real app, compare hashed password!
            elif username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username}!")
                navigate_to("dashboard") # Navigate immediately after success
            else:
                st.error("Invalid username or password.")

    if st.button("Go to Register"):
        navigate_to("register")


def dashboard_page():
    st.markdown("""
        <style>
        .stButton>button { width: 100%; height: 80px; font-size: 20px; margin-bottom: 15px; }
        .stButton>button:hover { background-color: #f0f0f0; color: black; border-color: #555; }
         /* Style for logout button */
        div.stButton > button[kind="secondary"] {
             background-color: #f63366; color: white; border: none;
        }
        div.stButton > button[kind="secondary"]:hover {
            background-color: #c70d3a; color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title(f"Dashboard - Welcome {st.session_state.username}")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Symptom Analyzer"):
            navigate_to("symptom_analyzer")
    with col2:
        if st.button("Brain Tumor Analyzer"):
            navigate_to("brain_tumor_analyzer")
    with col3:
        if st.button("AI Chatbot"):
            navigate_to("chatbot")

    st.markdown("---")
    # Add Logout button
    if st.button("Logout", type="secondary"): # Use type for different styling potential
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = "main"
        # Clear chat history if desired on logout
        if "chat_session" in st.session_state:
            del st.session_state["chat_session"]
        if "messages" in st.session_state: # If using a simple list for chat history
             del st.session_state["messages"]
        st.experimental_rerun()


def symptom_analyzer_page():
    st.markdown("<h1 style='text-align: center;'>Symptom Analyser</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("This tool is intended for informational purposes only and does not substitute professional medical advice.")

    html_file_path = 'index.html'
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=600, scrolling=True)
    except FileNotFoundError:
        st.error(f"Error: {html_file_path} not found. Make sure it's in the project directory.")
    except Exception as e:
        st.error(f"Error loading the symptom analyzer: {e}")

    st.markdown("---")
    if st.button("⬅ Back to Dashboard"):
        navigate_to("dashboard")


def brain_tumor_analyzer_page():
    st.title("🧠 Brain Tumor Analyzer")
    st.markdown("Upload an MRI image (.jpg, .jpeg, .png) to classify potential tumor types.")
    st.warning("⚠ Disclaimer: This AI tool provides a preliminary analysis based on image patterns. It is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB') # Ensure image is RGB

            # Display uploaded image
            max_width = 350
            img_aspect_ratio = image.width / image.height
            st.image(image, caption="Uploaded MRI Image", width=max_width)

            # Classify button
            if st.button("Analyze Image"):
                with st.spinner('Analyzing...'):
                    predicted_class, confidence = predict_image(image, alexnet)

                if predicted_class and confidence is not None:
                    st.success(f"Analysis Complete!")
                    st.markdown(f"*Predicted Class:* {predicted_class}")
                    st.markdown(f"*Confidence:* {confidence*100:.2f}%")

                    # Get and display assessment details
                    assessment = assess_tumor_emergency(predicted_class)
                    st.markdown("---")
                    st.markdown(f"### Assessment for {predicted_class}")
                    st.markdown(f"🚨 Emergency Level:** {assessment['emergency_level']}")
                    st.markdown(f"🩺 Recommended Action:** {assessment['action']}")
                    st.markdown(f"🧬 Origin:** {assessment['origin']}")
                    st.markdown(f"ℹ Description:")
                    st.info(f"{assessment['description']}") # Use st.info for description box
                else:
                    st.error("Could not classify the image. Please try another image or check logs.")

        except Exception as e:
            st.error(f"An error occurred processing the image: {e}")

    st.markdown("---")
    if st.button("⬅ Back to Dashboard"):
        navigate_to("dashboard")


import streamlit as st

def chatbot_page():
    st.title("💬 AI Medical Chatbot")
    st.markdown(
        "Ask medical questions, get information, or find nearby hospitals. "
        "Remember, this chatbot provides information and is not a substitute for professional medical advice."
    )
    st.markdown("---")

    # Initialize chat history
    if "chat_session" not in st.session_state:
        try:
            st.session_state.chat_session = chat_model.start_chat(history=[])
        except Exception as e:
            st.error(f"Failed to start chat session: {e}")
            st.button("⬅ Back to Dashboard", on_click=navigate_to, args=("dashboard",))
            return  # Stop rendering the rest of the page

    # Display chat messages from history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Accept user input
    user_input = st.text_input("Ask something...", placeholder="Type your question here...")

    if user_input:
        # Build detailed prompt for medical assistant context
        user_prompt = f"""
You are a helpful and professional medical assistant AI. Provide clear, concise, and accurate medical information.
Always remind users that your advice is not a substitute for professional medical diagnosis or treatment.
Encourage users to see a healthcare professional for serious or urgent issues.

Answer the following question carefully and politely:

{user_input}
"""

        # Add user message to chat history and display it
        st.chat_message("user").markdown(user_input)

        # Send user message to Gemini and get response
        try:
            with st.spinner("Thinking..."):
                gemini_response = st.session_state.chat_session.send_message(user_prompt)

            # Display Gemini response
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)

        except Exception as e:
             st.error(f"An error occurred communicating with the AI: {e}")
             # Optionally add a retry or info message to the chat history
             st.chat_message("assistant").markdown(f"Sorry, I encountered an error: {e}")


    st.markdown("---") # Separator before the back button
    # Use on_click for button navigation
    st.button("⬅ Back to Dashboard", on_click=navigate_to, args=("dashboard",))


# --- Main App Logic ---

# Check login status for protected pages
if st.session_state.page not in ["main", "register", "signin"] and not st.session_state.logged_in:
    st.warning("Please log in to access this page.")
    st.session_state.page = "signin" # Redirect to signin if not logged in

# Page Routing
if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "signin":
    signin_page()
elif st.session_state.page == "dashboard":
    if st.session_state.logged_in:
        dashboard_page()
    else: # Should not happen due to check above, but as fallback
        navigate_to("signin")
elif st.session_state.page == "symptom_analyzer":
     if st.session_state.logged_in:
        symptom_analyzer_page()
     else:
        navigate_to("signin")
elif st.session_state.page == "brain_tumor_analyzer":
     if st.session_state.logged_in:
        brain_tumor_analyzer_page()
     else:
        navigate_to("signin")
elif st.session_state.page == "chatbot":
     if st.session_state.logged_in:
        chatbot_page() # Renamed function
     else:
        navigate_to("signin")
else:
    # Fallback to main page if state is invalid
    st.session_state.page = "main"
    main_page()


# --- END OF FILE main.py ---