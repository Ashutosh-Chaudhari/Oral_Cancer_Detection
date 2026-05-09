import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import io
from clinical_predict import predict_clinical

# Load models once at module level
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Intraoral model
intraoral_model = resnet18()
intraoral_model.fc = nn.Linear(intraoral_model.fc.in_features, 2)
intraoral_model.load_state_dict(torch.load('backend/models/intraoral_resnet18.pth', map_location=device))
intraoral_model = intraoral_model.to(device)
intraoral_model.eval()

# Histopath model
histopath_model = resnet18()
histopath_model.fc = nn.Linear(histopath_model.fc.in_features, 2)
histopath_model.load_state_dict(torch.load('backend/models/histopath_resnet18.pth', map_location=device))
histopath_model = histopath_model.to(device)
histopath_model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_and_predict(image_file, model):
    """Preprocess image and get model prediction"""
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        cancer_prob = probabilities[0][1].item()  # Probability of cancer class
    
    return cancer_prob

def predict(intraoral_image, histopath_image, clinical_data):
    # Get predictions from both image models
    intraoral_score = preprocess_and_predict(intraoral_image, intraoral_model)
    histopath_score = preprocess_and_predict(histopath_image, histopath_model)
    
    # Get clinical prediction
    clinical_result = predict_clinical(clinical_data)
    clinical_score = clinical_result["clinical_score"] / 100  # Normalize to 0-1
    
    # Weighted fusion
    final_score = (
        0.4 * intraoral_score +
        0.3 * histopath_score +
        0.3 * clinical_score
    )
    
    # Agreement rule: if 2+ models predict high risk (>0.7), boost final score
    scores = [intraoral_score, histopath_score, clinical_score]
    if sum(s > 0.7 for s in scores) >= 2:
        final_score = max(final_score, 0.75)
    
    # Final classification
    if final_score < 0.3:
        final_risk = "Low"
    elif final_score < 0.7:
        final_risk = "Medium"
    else:
        final_risk = "High"
    
    # Build recommendation
    if final_risk == "High":
        recommendation = "Urgent referral to oncologist recommended"
    elif final_risk == "Medium":
        recommendation = "Follow-up consultation recommended within 2 weeks"
    else:
        recommendation = "Continue routine monitoring"
    
    # Feature importance from clinical data
    feature_importance = {}
    if clinical_data.get("tobacco", 0) == 1:
        feature_importance["Tobacco Use"] = 25
    if clinical_data.get("smoking", 0) == 1:
        feature_importance["Smoking"] = 20
    if clinical_data.get("alcohol", 0) == 1:
        feature_importance["Alcohol"] = 15
    if clinical_data.get("age", 0) > 50:
        feature_importance["Age"] = 15
    if clinical_data.get("family_history", 0) == 1:
        feature_importance["Family History"] = 10
    
    response = {
        "intraoral_score": round(intraoral_score * 100, 2),
        "histopath_score": round(histopath_score * 100, 2),
        "clinical_score": round(clinical_score * 100, 2),
        "final_score": round(final_score * 100, 2),
        "final_risk": final_risk,
        "risk_score": round(final_score * 100, 2),  # For backward compatibility
        "label": final_risk,  # For backward compatibility
        "image_score": round((intraoral_score + histopath_score) / 2 * 100, 2),  # Average of both
        "weights": {
            "intraoral": 0.4,
            "histopath": 0.3,
            "clinical": 0.3
        },
        "confidence": max(intraoral_score, histopath_score, clinical_score),
        "heatmap_url": "/static/heatmap.jpg",
        "recommendation": recommendation,
        "summary": f"{final_risk} risk detected based on image analysis and clinical factors",
        "fusion_summary": "Final risk computed using weighted combination of intraoral (40%), histopath (30%), and clinical (30%) inputs",
        "feature_importance": feature_importance
    }
    return response
