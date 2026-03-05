"""
Full hybrid scoring diagnostic — verifies ML + rule-based combined output.
"""
import warnings; warnings.filterwarnings("ignore")
import joblib, json, numpy as np, os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
risk_model = joblib.load(os.path.join(MODEL_DIR, "risk_progression_model.pkl"))
scaler_cls = joblib.load(os.path.join(MODEL_DIR, "scaler_classification.pkl"))

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLS = json.load(f)

def encode(age, screen, nearw, outdoor, parents=2, fam=1, tuition=1, comp=1,
           sports=0, school=1, vitd=0, sex=1, height=140, weight=35):
    bmi = weight / ((height/100)**2)
    bmi_cat = 0 if bmi<18.5 else 1 if bmi<25 else 2 if bmi<30 else 3
    _out_f = max(outdoor, 0.5)
    de = min(screen / (_out_f + 0.1), 20.0)
    risk_sc = fam*2 + screen/2 + nearw/2 + (10-outdoor) + tuition + comp
    row = {
        "Age": age, "Sex": sex, "Height_cm": height, "Weight_kg": weight,
        "BMI": round(bmi,2), "BMI_Category": bmi_cat,
        "Location_Type": 1, "School_Type": school,
        "Family_History_Myopia": fam, "Parents_With_Myopia": parents,
        "Screen_Time_Hours": screen, "Near_Work_Hours": nearw,
        "Outdoor_Time_Hours": outdoor,
        "Tuition_Classes": tuition, "Competitive_Exam_Prep": comp,
        "Vitamin_D_Supplementation": vitd, "Sports_Participation": sports,
        "Screen_Near_Work": screen+nearw, "Outdoor_Activity_Role": outdoor*sports,
        "Digital_Exposure": de, "Academic_Stress": tuition*comp,
        "Risk_Score": min(risk_sc, 20), "State_Encoded": 5,
        "State_Delhi":0,"State_Gujarat":0,"State_Karnataka":0,"State_Kerala":0,
        "State_Maharashtra":1,"State_Punjab":0,"State_Rajasthan":0,
        "State_Tamil Nadu":0,"State_Telangana":0,
        "State_Uttar Pradesh":0,"State_West Bengal":0,
    }
    vals = [row.get(c, 0) for c in FEATURE_COLS]
    return np.array(vals, dtype=float).reshape(1,-1)

def ml_score(X):
    return risk_model.predict_proba(scaler_cls.transform(X))[0][1]

def rule_based(age, screen, nearw, outdoor, parents=2, fam=1, tuition=1, comp=1,
               sports=0, school=1, vitd=0):
    s = 30
    if age<=8: s+=15
    elif age<=10: s+=10
    elif age<=12: s+=5
    if parents==2: s+=25
    elif parents==1: s+=15
    elif fam: s+=8
    if screen>8: s+=22
    elif screen>6: s+=17
    elif screen>4: s+=12
    elif screen>2: s+=6
    if outdoor==0: s+=25
    elif outdoor<0.5: s+=20
    elif outdoor<1: s+=15
    elif outdoor<2: s+=8
    elif outdoor>=3: s-=10
    if nearw>6: s+=15
    elif nearw>4: s+=8
    if comp: s+=10
    if tuition: s+=5
    if school in (1,2): s+=3
    if vitd: s-=5
    if sports==2: s-=8
    return min(max(s, 0), 100) / 100.0

def hybrid(ml, rule):
    if ml >= 0.65:     p = 0.45*ml + 0.55*rule
    elif ml >= 0.35:   p = 0.25*ml + 0.75*rule
    else:              p = 0.10*ml + 0.90*rule
    return max(p, 0.80*rule)

cases = [
    # label, age,sc,nw,outdoor,parents,fam,tu,co,sp,sch,vd, expected
    ("A  outdoor=0.5  screen=8    → HIGH",    12, 8,5,0.5,2,1,1,1,0,1,0,"HIGH"),
    ("B  outdoor=0    screen=10   → HIGH",     8,10,8,0.0,2,1,1,1,0,1,0,"HIGH"),
    ("B2 outdoor=0.1  screen=10   → HIGH",     8,10,8,0.1,2,1,1,1,0,1,0,"HIGH"),
    ("B3 outdoor=0.3  screen=8    → HIGH",     9, 8,6,0.3,2,1,1,1,0,1,0,"HIGH"),
    ("C  outdoor=1    screen=8    → HIGH",    10, 8,5,1.0,2,1,1,1,0,1,0,"HIGH"),
    ("D  outdoor=2    screen=4    → LOW",     14, 4,2,2.0,0,0,0,0,2,0,0,"LOW"),
    ("E  outdoor=0.2  screen=9    → HIGH",     9, 9,7,0.2,2,1,1,1,0,1,0,"HIGH"),
    ("F  outdoor=3    screen=1    → LOW",     14, 1,1,3.0,0,0,0,0,2,0,1,"LOW"),
    ("G  outdoor=1 no-parents sc=2 → LOW",   13, 2,2,1.0,0,0,0,0,1,0,0,"LOW"),
    ("H  both-parents screen=6   → HIGH",    11, 6,4,1.0,2,1,0,0,1,1,0,"HIGH"),
    ("I  age=7 max indoor risk   → HIGH",     7,12,9,0.0,2,1,1,1,0,1,0,"HIGH"),
    ("J  age=15 moderate         → MOD",     15, 5,3,1.0,1,1,0,0,1,1,0,"MODERATE"),
]

print("=" * 90)
print("HYBRID SCORING — ML + Clinical Rule Combined")
print("=" * 90)
print(f"\n{'Label':42s} {'ML%':>6s} {'Rule%':>6s} {'Final%':>7s} {'Level':>10s}  OK?")
print("-" * 90)

all_ok = True
for row in cases:
    label=row[0]; params=row[1:-1]; exp=row[-1]
    age_,sc_,nw_,out_,par_,fa_,tu_,co_,sp_,sch_,vd_ = params
    X  = encode(age=age_,screen=sc_,nearw=nw_,outdoor=out_,parents=par_,fam=fa_,
                tuition=tu_,comp=co_,sports=sp_,school=sch_,vitd=vd_)
    ml = ml_score(X)
    rl = rule_based(age_,sc_,nw_,out_,par_,fa_,tu_,co_,sp_,sch_,vd_)
    fin= hybrid(ml, rl)
    lvl= "HIGH" if fin>=0.70 else "MODERATE" if fin>=0.40 else "LOW"
    ok = "✓" if lvl==exp else f"✗ (got {lvl})"
    if "✗" in ok: all_ok = False
    print(f"{label:42s} {ml*100:6.1f}% {rl*100:6.1f}% {fin*100:7.1f}%  {lvl:>10s}  {ok}")

print("\n" + ("✅  ALL CASES CORRECT" if all_ok else "⚠️  SOME CASES NEED REVIEW"))

risk_model = joblib.load(os.path.join(MODEL_DIR, "risk_progression_model.pkl"))
scaler_cls = joblib.load(os.path.join(MODEL_DIR, "scaler_classification.pkl"))

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLS = json.load(f)

def encode(age, screen, nearw, outdoor, parents=2, fam=1, tuition=1, comp=1, sports=0,
           school=1, vitd=0, sex=1, height=140, weight=35):
    bmi = weight / ((height/100)**2)
    bmi_cat = 0 if bmi<18.5 else 1 if bmi<25 else 2 if bmi<30 else 3
    _out_f = max(outdoor, 0.5)
    de = min(screen / (_out_f + 0.1), 20.0)
    risk_sc = fam*2 + screen/2 + nearw/2 + (10-outdoor) + tuition + comp
    
    row = {
        "Age": age, "Sex": sex, "Height_cm": height, "Weight_kg": weight,
        "BMI": round(bmi,2), "BMI_Category": bmi_cat,
        "Location_Type": 1, "School_Type": school,
        "Family_History_Myopia": fam, "Parents_With_Myopia": parents,
        "Screen_Time_Hours": screen, "Near_Work_Hours": nearw,
        "Outdoor_Time_Hours": outdoor,
        "Tuition_Classes": tuition, "Competitive_Exam_Prep": comp,
        "Vitamin_D_Supplementation": vitd, "Sports_Participation": sports,
        "Screen_Near_Work": screen+nearw,
        "Outdoor_Activity_Role": outdoor*sports,
        "Digital_Exposure": de,
        "Academic_Stress": tuition*comp,
        "Risk_Score": min(risk_sc, 20),
        "State_Encoded": 5,
        "State_Delhi":0,"State_Gujarat":0,"State_Karnataka":0,"State_Kerala":0,
        "State_Maharashtra":1,"State_Punjab":0,"State_Rajasthan":0,
        "State_Tamil Nadu":0,"State_Telangana":0,"State_Uttar Pradesh":0,"State_West Bengal":0,
    }
    vals = [row.get(c, 0) for c in FEATURE_COLS]
    return np.array(vals, dtype=float).reshape(1,-1)

def ml_prob(X):
    Xs = scaler_cls.transform(X)
    return risk_model.predict_proba(Xs)[0][1]

print("=" * 72)
print("PROBLEM ANALYSIS: What the raw ML model returns vs expected")
print("=" * 72)
print(f"\n{'Label':40s} {'ML%':>6s}  {'Expected':>10s}  {'OK?'}")
print("-" * 72)

cases = [
    # (label, age, screen, nearw, outdoor, parents, fam, tuition, comp, sports)
    ("A  outdoor=0.5  screen=8  HIGH-RISK",   12,  8, 5, 0.5, 2, 1, 1, 1, 0),
    ("B  outdoor=0    screen=10 HIGH-RISK",    8, 10, 8, 0.0, 2, 1, 1, 1, 0),
    ("B2 outdoor=0.1  screen=10 HIGH-RISK",    8, 10, 8, 0.1, 2, 1, 1, 1, 0),
    ("C  outdoor=1    screen=8  HIGH-RISK",   10,  8, 5, 1.0, 2, 1, 1, 1, 0),
    ("D  outdoor=2    screen=4  LOW-RISK",    14,  4, 2, 2.0, 0, 0, 0, 0, 2),
    ("E  outdoor=0.2  screen=9  HIGH-RISK",    9,  9, 7, 0.2, 2, 1, 1, 1, 0),
    ("F  outdoor=3    screen=1  LOW-RISK",    14,  1, 1, 3.0, 0, 0, 0, 0, 2),
    ("G  outdoor=1    parents=0 screen=2 LOW", 13, 2, 2, 1.0, 0, 0, 0, 0, 1),
    ("H  both parents screen=6  outdoor=1",   11,  6, 4, 1.0, 2, 1, 0, 0, 1),
]
expected = ["HIGH","HIGH","HIGH","HIGH","LOW","HIGH","LOW","LOW","HIGH"]

for (case, *params), exp in zip(cases, expected):
    X = encode(*params)
    p = ml_prob(X)
    ok = "✓" if (exp=="HIGH" and p>0.4) or (exp=="LOW" and p<0.5) else "✗ BUG"
    print(f"{case:40s} {p*100:6.1f}%  {exp:>10s}  {ok}")

print("\n" + "=" * 72)
print("FEATURE RANGE ANALYSIS: outdoor_time sweep (all else HIGH-RISK)")
print("=" * 72)
print(f"\n{'outdoor_time':>14s}  {'DE_raw':>8s}  {'DE_capped':>10s}  {'ML prob':>8s}")
for out in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    _f = max(out, 0.5)
    de_raw = 8 / (out + 0.1) if out > 0 else 80.0
    de_cap = min(8 / (_f + 0.1), 20.0)
    X = encode(age=10, screen=8, nearw=5, outdoor=out)
    p = ml_prob(X)
    print(f"{out:14.1f}  {de_raw:8.2f}  {de_cap:10.2f}  {p*100:8.1f}%")
