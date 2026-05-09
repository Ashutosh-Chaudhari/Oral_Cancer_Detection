import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import io
import os
from gradcam import generate_gradcam

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file):
    try:
        img = Image.open(file)
        img.verify()
        return True
    except:
        return False

# Load models once at module level
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Intraoral model
intraoral_model = None
intraoral_model_path = "backend/models/intraoral_resnet18.pth"
if os.path.exists(intraoral_model_path):
    intraoral_model = models.resnet18(weights=None)
    intraoral_model.fc = nn.Linear(intraoral_model.fc.in_features, 2)
    intraoral_model.load_state_dict(torch.load(intraoral_model_path, map_location=device))
    intraoral_model.to(device)
    intraoral_model.eval()

# Histopath model
histopath_model = None
histopath_model_path = "backend/models/histopath_resnet18.pth"
if os.path.exists(histopath_model_path):
    histopath_model = models.resnet18(weights=None)
    histopath_model.fc = nn.Linear(histopath_model.fc.in_features, 2)
    histopath_model.load_state_dict(torch.load(histopath_model_path, map_location=device))
    histopath_model.to(device)
    histopath_model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(model, image_file):
    """Predict using trained model and return score with tensor"""
    if model is None:
        return 0.5, None, None
    
    image = Image.open(image_file)
    image = image.convert('RGB')
    image_np = np.array(image)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        cancer_prob = probs[0][0].item()
    
    return cancer_prob, image_tensor, image_np

def predict(intraoral_image, histopath_image, clinical_data):
    # Handle missing images
    if intraoral_image is None:
        intraoral_score = 0.0
    else:
        # Validate file type
        if not allowed_file(intraoral_image.filename):
            return {"error": "Invalid intraoral image format"}, 400
        
        # Validate image content
        if not validate_image(intraoral_image):
            return {"error": "Corrupted intraoral image"}, 400
        intraoral_image.seek(0)
        
        # Validate image size
        img = Image.open(intraoral_image)
        if img.size[0] < 100 or img.size[1] < 100:
            return {"error": "Intraoral image too small"}, 400
        intraoral_image.seek(0)
        
        # Predict
        intraoral_score, intraoral_tensor, intraoral_np = predict_image(intraoral_model, intraoral_image)
    
    if histopath_image is None:
        histopath_score = 0.0
        histopath_tensor = None
        histopath_np = None
    else:
        # Validate file type
        if not allowed_file(histopath_image.filename):
            return {"error": "Invalid histopath image format"}, 400
        
        # Validate image content
        if not validate_image(histopath_image):
            return {"error": "Corrupted histopath image"}, 400
        histopath_image.seek(0)
        
        # Validate image size
        img = Image.open(histopath_image)
        if img.size[0] < 100 or img.size[1] < 100:
            return {"error": "Histopath image too small"}, 400
        histopath_image.seek(0)
        
        # Predict
        histopath_score, histopath_tensor, histopath_np = predict_image(histopath_model, histopath_image)
    
    # Generate Grad-CAM heatmap for intraoral image if available
    heatmap_url = None
    if intraoral_model is not None and intraoral_tensor is not None:
        try:
            os.makedirs("static", exist_ok=True)
            target_layer = intraoral_model.layer4[-1]
            cam = generate_gradcam(intraoral_model, intraoral_tensor, target_layer)
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            original = cv2.resize(intraoral_np, (224, 224))
            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
            
            cv2.imwrite("static/heatmap.jpg", overlay)
            heatmap_url = "/static/heatmap.jpg"
        except:
            pass
    
    # Clinical scoring
    score = 0
    age = clinical_data.get("age", 0)
    tobacco = clinical_data.get("tobacco", 0)
    smoking = clinical_data.get("smoking", 0)
    alcohol = clinical_data.get("alcohol", 0)
    family_history = clinical_data.get("family_history", 0)
    oral_lesions = clinical_data.get("oral_lesions", 0)
    unexplained_bleeding = clinical_data.get("unexplained_bleeding", 0)
    difficulty_swallowing = clinical_data.get("difficulty_swallowing", 0)
    patches = clinical_data.get("patches", 0)
    
    score += tobacco * 0.20
    score += smoking * 0.18
    score += oral_lesions * 0.15
    score += alcohol * 0.12
    score += unexplained_bleeding * 0.10
    score += difficulty_swallowing * 0.08
    score += patches * 0.08
    score += family_history * 0.07
    
    if age > 50:
        score += 0.12
    elif age > 40:
        score += 0.08
    
    clinical_score = min(score, 1.0)
    
    # Weighted fusion
    final_score = (
        0.4 * intraoral_score +
        0.3 * histopath_score +
        0.3 * clinical_score
    )
    
    # Agreement rule
    scores = [intraoral_score, histopath_score, clinical_score]
    if sum(s > 0.7 for s in scores) >= 2:
        final_score = max(final_score, 0.75)
    
    # Classification
    if final_score < 0.3:
        final_risk = "Low"
    elif final_score < 0.7:
        final_risk = "Medium"
    else:
        final_risk = "High"
    
    # Recommendation
    if final_risk == "High":
        recommendation = "Urgent referral to oncologist recommended"
    elif final_risk == "Medium":
        recommendation = "Follow-up consultation recommended within 2 weeks"
    else:
        recommendation = "Continue routine monitoring"
    
    # Feature importance
    feature_importance = {}
    if tobacco == 1:
        feature_importance["Tobacco Use"] = 25
    if smoking == 1:
        feature_importance["Smoking"] = 20
    if alcohol == 1:
        feature_importance["Alcohol"] = 15
    if age > 50:
        feature_importance["Age"] = 15
    if family_history == 1:
        feature_importance["Family History"] = 10
    
    response = {
        "intraoral_score": round(intraoral_score * 100, 2),
        "histopath_score": round(histopath_score * 100, 2),
        "clinical_score": round(clinical_score * 100, 2),
        "final_score": round(final_score * 100, 2),
        "final_risk": final_risk,
        "weights": {
            "intraoral": 0.4,
            "histopath": 0.3,
            "clinical": 0.3
        },
        "confidence": max(intraoral_score, histopath_score, clinical_score),
        "recommendation": recommendation,
        "summary": f"{final_risk} risk detected based on image analysis and clinical factors",
        "fusion_summary": "Final risk computed using weighted combination of intraoral (40%), histopath (30%), and clinical (30%) inputs",
        "feature_importance": feature_importance,
        "heatmap_url": heatmap_url
    }
    return response
