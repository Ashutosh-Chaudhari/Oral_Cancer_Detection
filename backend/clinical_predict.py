def predict_clinical(data: dict):
    """
    Rule-based clinical risk scoring system
    """
    
    age = data.get("age", 0)
    gender = data.get("gender", 0)
    tobacco = data.get("tobacco", 0)
    smoking = data.get("smoking", 0)
    alcohol = data.get("alcohol", 0)
    family_history = data.get("family_history", 0)
    oral_lesions = data.get("oral_lesions", 0)
    unexplained_bleeding = data.get("unexplained_bleeding", 0)
    difficulty_swallowing = data.get("difficulty_swallowing", 0)
    patches = data.get("patches", 0)
    
    score = 0
    
    # High impact
    score += tobacco * 0.25
    score += oral_lesions * 0.20
    score += patches * 0.15
    
    # Medium impact
    score += smoking * 0.10
    score += alcohol * 0.10
    score += unexplained_bleeding * 0.10
    
    # Lower impact
    score += difficulty_swallowing * 0.05
    score += family_history * 0.05
    
    # Age factor
    if age > 50:
        score += 0.10
    
    score = min(score, 1.0)
    
    if score < 0.3:
        risk = "Low"
    elif score < 0.7:
        risk = "Medium"
    else:
        risk = "High"
    
    return {
        "clinical_score": round(score * 100, 2),
        "risk_level": risk
    }
