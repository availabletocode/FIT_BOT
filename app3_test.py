import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# =====================
# LOAD MODEL
# =====================
bundle = joblib.load("fitbot_model2_full_datatrain.pkl")
model = bundle["model"]
mlb = bundle["mlb"]
ref_df = bundle["reference_df"]

# =====================
# HELPERS
# =====================
def height_to_cm(h):
    f, i = map(int, h.split())
    return f * 30.48 + i * 2.54

def bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 24.9: return "Normal"
    elif bmi < 29.9: return "Overweight"
    else: return "Obesity"

def age_group(age):
    if 18 <= age <= 24: return "18-24"
    elif 25 <= age <= 34: return "25-34"
    elif 35 <= age <= 44: return "35-44"
    elif 45 <= age <= 54: return "45-54"
    else: return "55-60"

# =====================
# UI
# =====================
st.title("👩‍⚕️ FitBot — Health Advisor (ML Powered)")

name = st.text_input("Enter your name")
age = st.number_input("Age", min_value=10, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])

if age < 18 or age > 60:
    st.error(" Sorry FitBot is only for adults between 18 and 60 years old.")
    st.stop()

if gender == "Female":
    pregnant = st.selectbox("Are you pregnant?", ["No", "Yes"])
    if pregnant == "Yes":
        st.warning("🤰 Please consult a doctor.")
        st.stop()

height = st.text_input("Height (ft in)", "5 2")
weight = st.number_input("Weight (kg)", 30.0, 200.0)

if st.button("Check Health"):
    h_cm = height_to_cm(height)
    bmi = round(weight / ((h_cm / 100) ** 2), 1)
    bmi_cat = bmi_category(bmi)
    ag = age_group(age)

    st.success(f"{name}, your BMI is {bmi} ({bmi_cat})")

    bmi_fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=bmi,
    title={"text": "BMI Level"},
    gauge={
        "axis": {"range": [0, 40]},
        "steps": [
            {"range": [0, 18.5], "color": "#87CEEB"},     # sky blue    for   underweight
            {"range": [18.5, 24.9], "color": "#90EE90"},  # light green for   normal
            {"range": [25, 29.9], "color": "#FFD580"},    # soft orange for   overweight
            {"range": [30, 40], "color": "#FF7F7F"}       # red         for   obesity 
          ]
        }
    ))
    st.plotly_chart(bmi_fig, width="stretch")


    st.info(f"⚖️ Body Weight Status: **{bmi_cat}**")



    input_df = pd.DataFrame([{
        "Age_Group": ag,
        "Gender": gender,
        "BMI_Category": bmi_cat
    }])

    pred = model.predict(input_df)
    diseases = mlb.inverse_transform(pred)[0]

    row = ref_df[
        (ref_df["Age_Group"] == ag) &
        (ref_df["Gender"] == gender) &
        (ref_df["BMI_Category"] == bmi_cat)
    ].iloc[0]

    st.subheader("🩺 Possible Health Risks(diseases)")
    st.write(", ".join(diseases))

    st.subheader("💡 Health Advice")
    st.write(row["Health_Advice"])

    st.subheader("🥗 Recommended Diet")
    st.write(row["Diet_Plan"])

    st.subheader("⚠️ Risk Level")
    st.write(row["Risk_Level"])
    # =====================
    # ⚠️ RISK LEVEL GAUGE
    # =====================
    risk_map = {
    "Low": 30,
    "Medium": 60,
    "High": 90
    }

    risk_score = risk_map.get(row["Risk_Level"], 50)

    risk_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Health Risk Level"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "#90EE90"},   # Green
                {"range": [40, 70], "color": "#FFD580"}, # Yellow
                {"range": [70, 100], "color": "#FF7F7F"} # Red
            ]
        }
    ))

    st.plotly_chart(risk_fig, width="stretch")


    st.subheader("🍽️ Detailed Meal Plan")
    st.write(row["Detailed_Diet_Plan"])

    st.subheader("🧘 Yoga (15-30 min)")
    st.write(row["Exercise_Yoga_Plan"])

    st.subheader("🧘 Yoga/Exercise links")
    st.caption("⚠️ These exercise videos are for general wellness. Please consult a doctor if you have medical conditions.")
    st.write(row["Yoga_Exercise_Links"])

    #st.subheader("🧘 Yoga/Exercise Links")

#yoga_links = row["Yoga_Exercise_Links"].split("|")

#for i, link in enumerate(yoga_links, start=1):
    #st.markdown(f"👉 [Watch Yoga Video {i}]({link.strip()})")

    # ALL THESE JUST ABOVE COMMNET PART IS ALSO CAN USED FOR SHOWING LINKS like this we use these """ """ for mutli line comments under only some function to show on website 

    # Added final message
    st.info(" If you feel maximum problem or symptoms worsen, we recommend consulting a known doctor/specialist immediately before taking medical decisions. ")

    
    st.subheader("👩‍⚕️ Doctor (Indore)")
    st.caption(" Below are some well-known nutrition, weight-management, and lifestyle specialists in Indore. ")
    st.write("""
. **Dr. Neha Sharma** — Clinical Nutritionist (Weight loss/gain, BMI care)
  🔗 [Search on Google](https://r.search.yahoo.com/_ylt=AwrKDbzagF9pKAIAD6i7HAx.;_ylu=Y29sbwNzZzMEcG9zAzIEdnRpZAMEc2VjA3Ny/RV=2/RE=1769076187/RO=10/RU=https%3a%2f%2fwww.pushpanjalihospitalbpl.com%2fdoctordetail%2fAzk%3d/RK=2/RS=.SAyFsqgfED3RA.bCw4t6Hbqr7w-)             
             
. **Dr. Pooja Jain** — Dietitian & Lifestyle Coach (PCOS, Thyroid, Obesity)  
  🔗 [Search on Google](https://www.google.com/search?q=Dr+Pooja+Jain+Dietitian+Indore)  
                    
. **Dr. Ankit Verma** — General Physician (Diabetes, BP, Metabolic health)  
  🔗 [Search on Google](https://www.google.com/search?q=Dr+Ankit+Verma+Physician+Indore)        
                     
. **Dr. Ritu Agrawal** — Women's Health & Nutrition Expert  
  🔗 [Search on Google](https://r.search.yahoo.com/_ylt=AwrKCbBxi19pBQIAxwu7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Ny/RV=2/RE=1769078897/RO=10/RU=https%3a%2f%2fwww.justdial.com%2fIndore%2fDr-Ritu-Agarwal-Sharda-Opposite-Marriott-Hotel-Vijay-Nagar%2f0731PX731-X731-220602215712-E8F5_BZDET/RK=2/RS=p0Q0.CQzXI54vLUeqANOAebcTxs-)

. **CHL / Apollo Hospitals (Indore)** — Nutrition & Preventive Health Clinics
  🔗 [View on Google Maps](https://www.google.com/maps/search/Apollo+Hospital+Indore)
             
. **Dr. Anupama Singh** — Hepatology & Digestive Diseases  
  🔗 [Search on Google](https://www.google.com/search?q=Dr+Anupama+Singh+Hepatologist+Indore) 

. **Dr. Rakesh Verma** — Cardiologist  
  🔗 [Search on Google](https://www.google.com/search?q=Dr+Rakesh+Verma+Cardiologist+Indore)                        
""") 
    
    # we use ** ** to make text bold ** shrashti** 
    # we use [text](links) for clickable links 
    # we use * * for italic text 

    # Added final message
    #st.info(" If you feel maximum problem or symptoms worsen, we recommend consulting a known doctor/specialist immediately before taking medical decisions. ")
  


    # to run this "streamlit run app3_test.py"
    