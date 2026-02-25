"""
Pediatric Myopia Risk Prediction Dataset Generator
Generates 5000 synthetic Indian children records with medically realistic
epidemiological distributions.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Healthy"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


# ---------------------------------------------------------------------------
# 1. Demographics
# ---------------------------------------------------------------------------

ages = np.random.randint(6, 18, size=N)          # 6–17 inclusive
sexes = np.random.choice(["Male", "Female"], size=N, p=[0.51, 0.49])

# ---------------------------------------------------------------------------
# 2. Anthropometrics – age-realistic height & weight (Indian growth refs)
# ---------------------------------------------------------------------------

# Mean height by age (cm) – approximate Indian child growth reference
age_height_mean = {
    6:  115, 7:  120, 8:  125, 9:  130, 10: 135,
    11: 140, 12: 146, 13: 152, 14: 158, 15: 163, 16: 167, 17: 170
}
age_height_sd   = {a: 4.5 for a in range(6, 18)}

age_weight_mean = {
    6:  19,  7:  21,  8:  24,  9:  27,  10: 30,
    11: 34,  12: 38,  13: 43,  14: 48,  15: 53, 16: 57, 17: 60
}
age_weight_sd   = {a: 5.0 for a in range(6, 18)}

heights, weights = [], []
for age in ages:
    h = np.clip(np.random.normal(age_height_mean[age], age_height_sd[age]), 108, 178)
    w_mean = age_weight_mean[age]
    # slight sex-based difference handled via noise
    w = np.clip(np.random.normal(w_mean, age_weight_sd[age]), 15, 78)
    heights.append(round(h, 1))
    weights.append(round(w, 1))

heights = np.array(heights)
weights = np.array(weights)

height_m  = np.round(heights / 100, 3)
h_square  = np.round(height_m ** 2, 4)
bmis      = np.round(weights / h_square, 2)
bmi_cats  = [bmi_category(b) for b in bmis]

# ---------------------------------------------------------------------------
# 3. School level from age
# ---------------------------------------------------------------------------

def school_level(age: int) -> str:
    if age <= 9:
        return "Primary"
    elif age <= 12:
        return "Middle School"
    else:
        return "High School"

levels = [school_level(a) for a in ages]

# ---------------------------------------------------------------------------
# 4. Risk factors
# ---------------------------------------------------------------------------

# Family history (First_Relative): 20–25% base prevalence in India
first_relative = np.random.choice(["Yes", "No"], size=N, p=[0.22, 0.78])

# Electronic device usage – skew toward higher use for older children
device_options = ["Less than 3 h", "3-5 h", "5-8 h", ">8 h"]

def device_prob(age: int) -> list:
    if age <= 9:
        return [0.55, 0.28, 0.12, 0.05]
    elif age <= 12:
        return [0.30, 0.35, 0.25, 0.10]
    else:
        return [0.15, 0.30, 0.35, 0.20]

devices = np.array([
    np.random.choice(device_options, p=device_prob(a)) for a in ages
])

# Sunlight – inverse of screen time tendency
sunlight_options = ["Less than 1 h", "1-3 h", "3-5 h", ">5 h"]

def sunlight_prob(age: int) -> list:
    if age <= 9:
        return [0.10, 0.30, 0.38, 0.22]
    elif age <= 12:
        return [0.18, 0.38, 0.30, 0.14]
    else:
        return [0.28, 0.40, 0.22, 0.10]

sunlights = np.array([
    np.random.choice(sunlight_options, p=sunlight_prob(a)) for a in ages
])

# ---------------------------------------------------------------------------
# 5. Refractive error – conditional probability model
# ---------------------------------------------------------------------------

# Base myopia prevalence driven by age + risk factors
# Target overall: 25–30 % myopia, 5–8 % hyperopia, 5–10 % astigmatism, rest none

re_types = []
re_degrees = []
diag_ages = []
correction_methods = []
laser_surgeries = []
presence_list = []

# Correction probabilities given type
def correction_prob(re_type: str, degree_val: float) -> str:
    if re_type == "No error":
        return "None"
    # High myopia → almost always spectacles/contacts
    abs_deg = abs(degree_val) if degree_val else 0
    if re_type == "Myopia":
        if abs_deg >= 4.0:
            return np.random.choice(
                ["Spectacles", "Contact Lenses", "Atropine"],
                p=[0.68, 0.18, 0.14]
            )
        else:
            return np.random.choice(
                ["Spectacles", "Contact Lenses", "Atropine", "None"],
                p=[0.65, 0.10, 0.10, 0.15]
            )
    elif re_type == "Hyperopia":
        return np.random.choice(
            ["Spectacles", "None"], p=[0.70, 0.30]
        )
    else:  # Astigmatism
        return np.random.choice(
            ["Spectacles", "Contact Lenses", "None"],
            p=[0.65, 0.15, 0.20]
        )


for i in range(N):
    age = int(ages[i])
    fr  = first_relative[i]
    dev = devices[i]
    sun = sunlights[i]

    # ---- Myopia probability ----
    base_myopia = 0.07 if age <= 9 else (0.19 if age <= 12 else 0.34)
    if fr == "Yes":
        base_myopia = min(base_myopia * 1.45, 0.62)
    if dev in ["5-8 h", ">8 h"]:
        base_myopia = min(base_myopia * 1.22, 0.65)
    if sun in ["Less than 1 h", "1-3 h"]:
        base_myopia = min(base_myopia * 1.15, 0.65)

    # ---- Hyperopia probability ----
    base_hyper = 0.065

    # ---- Astigmatism probability ----
    base_astig = 0.07

    # ---- None probability ----
    # Normalise so total = 1
    total = base_myopia + base_hyper + base_astig
    if total >= 1.0:
        scale = 0.95 / total
        base_myopia *= scale; base_hyper *= scale; base_astig *= scale; total = 0.95
    base_none = 1.0 - total

    probs = [base_myopia, base_hyper, base_astig, base_none]
    re_type = np.random.choice(["Myopia", "Hyperopia", "Astigmatism", "No error"], p=probs)

    # ---- Degree ----
    if re_type == "Myopia":
        presence = "Yes"
        # Age-dependent severity: younger → milder
        if age <= 9:
            deg = round(np.random.uniform(-0.50, -2.50), 2)
        elif age <= 12:
            deg = round(np.random.uniform(-0.50, -4.50), 2)
        else:
            # Allow up to -8.00 D but tail very thin (high myopia <5% of total)
            # Use truncated-ish distribution
            raw = np.random.exponential(1.5)
            deg = round(max(-8.00, -(0.50 + raw * 2.0)), 2)
            deg = max(deg, -8.00)
            deg = min(deg, -0.50)
        degree_str = str(deg) + " D"

        # Diagnosis age: must be ≤ current age
        # High myopia → earlier diagnosis
        abs_deg = abs(deg)
        earliest = max(4, age - int(abs_deg * 1.5 + 1))
        diag_age = np.random.randint(max(4, earliest), age + 1)

    elif re_type == "Hyperopia":
        presence = "Yes"
        deg = round(np.random.uniform(0.50, 4.00), 2)
        degree_str = "+" + str(deg) + " D"
        diag_age = np.random.randint(max(4, age - 5), age + 1)

    elif re_type == "Astigmatism":
        presence = "Yes"
        deg = round(np.random.uniform(0.50, 3.00), 2)
        degree_str = "+" + str(deg) + " D (cyl)"
        diag_age = np.random.randint(max(4, age - 5), age + 1)

    else:
        presence = "No"
        degree_str = "N/A"
        diag_age = None

    # ---- Correction ----
    corr = correction_prob(re_type, deg if re_type == "Myopia" else 0)

    # ---- Laser surgery – essentially never in children ----
    laser = np.random.choice(["No", "Yes"], p=[0.998, 0.002])

    re_types.append(re_type)
    re_degrees.append(degree_str)
    diag_ages.append(str(diag_age) if diag_age is not None else "N/A")
    correction_methods.append(corr)
    laser_surgeries.append(laser)
    presence_list.append(presence)

# ---------------------------------------------------------------------------
# 6. Assemble DataFrame
# ---------------------------------------------------------------------------

df = pd.DataFrame({
    "Presence_of_RE":    presence_list,
    "Sex":               sexes,
    "Age":               ages,
    "Height":            heights,
    "Weight":            weights,
    "BMI_Category":      bmi_cats,
    "Medical_Level":     levels,
    "First_Relative":    first_relative,
    "Electronic_Devices": devices,
    "Sunlight":          sunlights,
    "Type_of_RE":        re_types,
    "Degree_RE":         re_degrees,
    "Diagnosis_Age":     diag_ages,
    "Correction_Method": correction_methods,
    "Laser_Surgery":     laser_surgeries,
    "Height_Meter":      height_m,
    "H_Square":          h_square,
    "BMI":               bmis,
})

# ---------------------------------------------------------------------------
# 7. Validation summary
# ---------------------------------------------------------------------------

total      = len(df)
myopia_n   = (df["Type_of_RE"] == "Myopia").sum()
hyper_n    = (df["Type_of_RE"] == "Hyperopia").sum()
astig_n    = (df["Type_of_RE"] == "Astigmatism").sum()
none_n     = (df["Type_of_RE"] == "No error").sum()

high_myopia = df[df["Type_of_RE"] == "Myopia"]["Degree_RE"].apply(
    lambda x: float(x.replace(" D", "")) <= -6.00
).sum()

print("=" * 50)
print("Dataset Validation Summary")
print("=" * 50)
print(f"Total records       : {total}")
print(f"Myopia              : {myopia_n} ({100*myopia_n/total:.1f}%)")
print(f"Hyperopia           : {hyper_n}  ({100*hyper_n/total:.1f}%)")
print(f"Astigmatism         : {astig_n}  ({100*astig_n/total:.1f}%)")
print(f"No refractive error : {none_n} ({100*none_n/total:.1f}%)")
print(f"High myopia (≥-6 D) : {high_myopia} ({100*high_myopia/total:.1f}%)")
print(f"Family Hx Yes       : {(df['First_Relative']=='Yes').sum()}")
print(f"Laser=Yes (rare)    : {(df['Laser_Surgery']=='Yes').sum()}")
print("=" * 50)

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------

out_path = r"d:\development\workspace\Mayopia\Synthetic_Myopia_Dataset.csv"
df.to_csv(out_path, index=False)
print(f"Saved → {out_path}")
