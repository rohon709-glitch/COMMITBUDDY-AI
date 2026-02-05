import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# =============================
# LOAD ENV
# =============================
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SERPER_API_KEY or not GROQ_API_KEY:
    st.error("Missing API keys. Add SERPER_API_KEY and GROQ_API_KEY.")
    st.stop()

search_tool = SerperDevTool()

# =============================
# CUSTOM PREMIUM UI STYLE
# =============================
def apply_custom_style():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    h1 { font-weight: 600; }

    .stTabs [data-baseweb="tab-list"] { gap: 25px; }

    .stTabs [aria-selected="true"] {
        color: #FF4B4B !important;
        border-bottom: 2px solid #FF4B4B;
    }

    .stTextInput input,
    .stNumberInput input,
    textarea {
        background-color: #1A1D24 !important;
        color: white !important;
        border-radius: 10px !important;
    }

    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 12px 25px;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color: #FF2E2E;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================
# LLM USING GROQ
# =============================
def get_llm():
    return LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=GROQ_API_KEY
    )

# =============================
# CREATE AGENTS
# =============================
def create_agents():
    llm = get_llm()

    nutritionist = Agent(
        role="Nutrition Specialist",
        goal="Develop personalized nutrition recommendations",
        backstory="Expert in nutrition science",
        tools=[search_tool],
        llm=llm
    )

    medical_specialist = Agent(
        role="Medical Nutrition Therapist",
        goal="Analyze medical conditions and diet restrictions",
        backstory="Clinical nutrition expert",
        tools=[search_tool],
        llm=llm
    )

    diet_planner = Agent(
        role="Therapeutic Diet Planner",
        goal="Create enjoyable meal plans",
        backstory="Transforms clinical advice into meals",
        llm=llm
    )

    return nutritionist, medical_specialist, diet_planner

# =============================
# CREATE TASKS
# =============================
def create_tasks(nutritionist, medical_specialist, diet_planner, user_info):

    demographics = Task(
        description=f"""
        Age: {user_info['age']}
        Gender: {user_info['gender']}
        Height: {user_info['height']}
        Weight: {user_info['weight']}
        Activity Level: {user_info['activity_level']}
        Goals: {user_info['goals']}
        """,
        agent=nutritionist,
        expected_output="Detailed calorie requirements and nutrition breakdown"
    )

    medical = Task(
        description=f"""
        Conditions: {user_info['medical_conditions']}
        Medications: {user_info['medications']}
        Allergies: {user_info['allergies']}
        """,
        agent=medical_specialist,
        context=[demographics],
        expected_output="Medical diet restrictions and safe food recommendations"
    )

    diet_plan = Task(
        description=f"""
        Food Preferences: {user_info['food_preferences']}
        Cooking Ability: {user_info['cooking_ability']}
        Budget: {user_info['budget']}
        Cultural Factors: {user_info['cultural_factors']}
        """,
        agent=diet_planner,
        context=[demographics, medical],
        expected_output="Complete personalized diet meal plan"
    )

    return [demographics, medical, diet_plan]


# =============================
# RUN CREW
# =============================
def run_nutrition_advisor(user_info):
    nutritionist, medical_specialist, diet_planner = create_agents()
    tasks = create_tasks(nutritionist, medical_specialist, diet_planner, user_info)

    crew = Crew(
        agents=[nutritionist, medical_specialist, diet_planner],
        tasks=tasks
    )

    return crew.kickoff()

# =============================
# STREAMLIT APP
# =============================
def app():
    st.set_page_config(page_title="CommitBuddy AI", page_icon="🥗", layout="wide")
    apply_custom_style()

    # Header
    st.markdown("""
    <h1>🥗 CommitBuddy AI</h1>
    <p style='color:gray;'>Created by Rohan Jamader</p>
    <p style='color:gray;'>Personalized Nutrition & Commitment Assistant</p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Basic Information",
        "Health Details",
        "Preferences & Lifestyle"
    ])

    # ---------------- BASIC INFO ----------------
    with tab1:
        age = st.number_input("Age", 1, 120, 25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height = st.text_input("Height", "5'10\"")
        weight = st.text_input("Weight", "160 lbs")

        activity_level = st.radio(
            "Activity Level",
            ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
            horizontal=True
        )

        goals = st.multiselect(
            "Goals",
            ["Weight Loss", "Weight Gain", "Maintenance", "Muscle Building", "General Health"]
        )

    # ---------------- HEALTH ----------------
    with tab2:
        medical_conditions = st.text_area("Medical Conditions")
        medications = st.text_area("Medications")
        allergies = st.text_area("Allergies")

    # ---------------- LIFESTYLE ----------------
    with tab3:
        food_preferences = st.text_area("Food Preferences")
        cooking_ability = st.select_slider(
            "Cooking Ability",
            ["Very Limited", "Basic", "Average", "Advanced"]
        )

        budget = st.select_slider(
            "Budget",
            ["Very Limited", "Moderate", "Flexible"]
        )

        cultural_factors = st.text_area("Cultural Factors")

    # ---------------- USER DATA ----------------
    user_info = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "activity_level": activity_level,
        "goals": ", ".join(goals) if goals else "General Health",
        "medical_conditions": medical_conditions or "None",
        "medications": medications or "None",
        "allergies": allergies or "None",
        "food_preferences": food_preferences or "None",
        "cooking_ability": cooking_ability,
        "budget": budget,
        "cultural_factors": cultural_factors or "None"
    }

    # ---------------- BUTTON ----------------
    if st.button("Generate Nutrition Plan"):

        with st.spinner("🥗 Creating your personalized nutrition plan..."):
            result = run_nutrition_advisor(user_info)

        st.success("Plan Generated!")
        st.markdown(result)

# =============================
# RUN
# =============================
if __name__ == "__main__":
    app()
