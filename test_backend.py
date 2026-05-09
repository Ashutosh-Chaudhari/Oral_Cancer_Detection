import requests

# Test with sample data
files = {
    'intraoral_image': open('data/intraoral/val/cancer/002.jpeg', 'rb'),
    'histopath_image': open('data/intraoral/val/cancer/007.jpeg', 'rb')
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

try:
    response = requests.post('http://127.0.0.1:5000/predict', files=files, data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
finally:
    files['intraoral_image'].close()
    files['histopath_image'].close()
