import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("Diet2_with_plan_gender_age_specific_healthly_dataset.csv")
df.columns = df.columns.str.strip()

# =====================
# FEATURES & TARGETS
# =====================
X = df[["Age_Group", "Gender", "BMI_Category"]]

# MULTI-LABEL target
df["Possible_Diseases"] = df["Possible_Diseases"].apply(
    lambda x: [d.strip() for d in str(x).split(",")]
)

mlb = MultiLabelBinarizer()
y_disease = mlb.fit_transform(df["Possible_Diseases"])

# =====================
# PREPROCESSING
# =====================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)
    ]
)

# =====================
# MODEL
# =====================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", OneVsRestClassifier(rf))
])

# =====================
# TRAIN
# =====================
model.fit(X, y_disease)

# =====================
# SAVE EVERYTHING
# =====================
joblib.dump(
    {
        "model": model,
        "mlb": mlb,
        "reference_df": df
    },
    "fitbot_model2_full_datatrain.pkl"
)

print("✅ Model trained & saved as fitbot_model2_full_datatrain.pkl")
