import os
import subprocess
import sys

def setup_project():
    print("🚀 Setting up Emotion Detection Project...")
    
    # Check and create directories
    directories = ['data', 'model', 'static/css', 'static/js', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create sample data
    print("📊 Creating sample data...")
    try:
        subprocess.run([sys.executable, 'create_sample_data.py'], check=True)
    except:
        print("⚠️ Could not create sample data, but continuing...")
    
    # Train model
    print("🤖 Training model...")
    try:
        subprocess.run([sys.executable, 'model/train_direct.py'], check=True, cwd='.')
    except:
        print("⚠️ Could not train model, but continuing...")
    
    print("🎉 Setup completed! Run: python app.py")

if __name__ == "__main__":
    setup_project()