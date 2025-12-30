import pandas as pd
import os

def create_sample_dataset():
    """Create sample emotion dataset in multiple formats"""
    
    # Sample data
    sample_data = [
        # joy
        ("joy", "I am extremely happy and excited about this wonderful news"),
        ("joy", "This is the best day ever I feel amazing"),
        ("joy", "Feeling joyful and content with life"),
        ("joy", "Happiness is overflowing in my heart"),
        ("joy", "Everything is perfect and wonderful"),
        
        # sadness
        ("sadness", "Feeling very sad and lonely today everything seems hopeless"),
        ("sadness", "I feel so depressed and miserable right now"),
        ("sadness", "Tears are flowing Im so heartbroken"),
        ("sadness", "Deep sadness is consuming me"),
        ("sadness", "I feel empty and sad inside"),
        
        # anger
        ("anger", "This makes me so angry I could scream at someone"),
        ("anger", "This injustice makes me furious and resentful"),
        ("anger", "This makes my blood boil Im so mad"),
        ("anger", "Rage is building up inside me"),
        ("anger", "This is so frustrating and irritating"),
        
        # love
        ("love", "I love you so much you mean everything to me"),
        ("love", "You are the love of my life I adore you"),
        ("love", "My heart is full of love and affection"),
        ("love", "I cherish every moment with you"),
        ("love", "My love for you grows stronger"),
        
        # fear
        ("fear", "I am terrified and scared of what might happen next"),
        ("fear", "Im so anxious and fearful about the future"),
        ("fear", "Im scared and panicked about this situation"),
        ("fear", "Fear is overwhelming me completely"),
        ("fear", "Im filled with dread and fear"),
        
        # surprise
        ("surprise", "Wow that was completely unexpected and surprising"),
        ("surprise", "That was an incredible surprise thank you"),
        ("surprise", "What an amazing surprise I cant believe it"),
        ("surprise", "That was shockingly unexpected"),
        ("surprise", "This surprise made my day perfect")
    ]
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Create CSV format
    df_csv = pd.DataFrame(sample_data, columns=['label', 'text'])
    df_csv.to_csv('data/emotions.csv', index=False)
    print("✅ Created: data/emotions.csv")
    
    # Create TXT formats (train, test, val)
    train_data = sample_data[:18]  # 60% for training
    test_data = sample_data[18:24] # 20% for testing
    val_data = sample_data[24:]    # 20% for validation
    
    # Save train.txt
    with open('data/train.txt', 'w', encoding='utf-8') as f:
        for label, text in train_data:
            f.write(f"{label}\t{text}\n")
    print("✅ Created: data/train.txt")
    
    # Save test.txt
    with open('data/test.txt', 'w', encoding='utf-8') as f:
        for label, text in test_data:
            f.write(f"{label}\t{text}\n")
    print("✅ Created: data/test.txt")
    
    # Save val.txt
    with open('data/val.txt', 'w', encoding='utf-8') as f:
        for label, text in val_data:
            f.write(f"{label}\t{text}\n")
    print("✅ Created: data/val.txt")
    
    print(f"\n📊 Dataset created successfully!")
    print(f"📝 Total samples: {len(sample_data)}")
    print(f"🎯 Training samples: {len(train_data)}")
    print(f"🧪 Testing samples: {len(test_data)}")
    print(f"📋 Validation samples: {len(val_data)}")
    
    # Show label distribution
    labels = [item[0] for item in sample_data]
    from collections import Counter
    print("\n📈 Label distribution:")
    for label, count in Counter(labels).items():
        print(f"  {label}: {count}")

if __name__ == "__main__":
    create_sample_dataset()