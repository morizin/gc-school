# Waste Segregation ML Project

## Overview
Automated waste classification using deep learning with ResNet152V2 neural network.

## Features
- Image-based waste classification
- Three categories: Recyclable, Organic, Non-Recyclable
- Web interface using Streamlit

## Prerequisites
- Python 3.8+
- TensorFlow 2.16.1
- Streamlit

## Installation
```bash
git clone https://github.com/yourusername/waste-segregation.git
cd waste-segregation
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

## Model Details
- Architecture: ResNet152V2
- Input Shape: 224x224x3
- Classes: Recyclable, Organic, Non-Recyclable

## Contributing
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## License
MIT License