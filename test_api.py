import requests
import os

# Test images
intraoral_img = "data/intraoral/val/cancer/002.jpeg"
histopath_img = "data/intraoral/val/cancer/007.jpeg"  # Using intraoral as histopath is empty

# Test data
files = {
    'intraoral_image': open(intraoral_img, 'rb'),
    'histopath_image': open(histopath_img, 'rb')
}

data = {
    'age': 55,
    'gender': 1,
    'tobacco': 1,
    'smoking': 1,
    'alcohol': 1,
    'family_history': 0,
    'oral_lesions': 1,
    'unexplained_bleeding': 1,
    'difficulty_swallowing': 0,
    'patches': 1
}

print("Testing API endpoint...")
print(f"Intraoral image: {intraoral_img}")
print(f"Histopath image: {histopath_img}")
print(f"Clinical data: {data}\n")

try:
    response = requests.post('http://127.0.0.1:5000/predict', files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS!")
        print(f"\nIntraoral Score: {result['intraoral_score']}%")
        print(f"Histopath Score: {result['histopath_score']}%")
        print(f"Clinical Score: {result['clinical_score']}%")
        print(f"Final Score: {result['final_score']}%")
        print(f"Risk Level: {result['final_risk']}")
        print(f"Recommendation: {result['recommendation']}")
    else:
        print(f"❌ FAILED: {response.status_code}")
        print(response.json())
except Exception as e:
    print(f"❌ ERROR: {e}")
finally:
    files['intraoral_image'].close()
    files['histopath_image'].close()
