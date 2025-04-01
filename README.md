# QuantumMMM

A tool that helps understand how different marketing activities affect sales.

![Version](https://img.shields.io/badge/version-0.1.0-blue)

## What This Project Does

QuantumMMM helps marketers answer questions like:
- Which marketing channels work best?
- How much money should I spend on each channel?
- How long do marketing effects last?
- What happens when I increase or decrease my marketing budget?

## Main Features

### Data Tools
- Create test data that looks like real marketing data
- Add realistic patterns like seasonality and special promotions
- Set different effectiveness levels for each marketing channel

### Marketing Effects Analysis
- See how marketing effects carry over to future time periods
- Account for diminishing returns (when more spending brings smaller gains)
- Visualize how different channels affect sales

### Modeling Options
- Use different types of statistical models
- Compare model performance to find the best one
- See which marketing channels have the biggest impact

### Results & Insights
- Break down sales by marketing channel
- Calculate ROI (return on investment) for each channel
- Find the best way to split your marketing budget

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/QuantumMMM.git
cd QuantumMMM
```

2. Create a virtual environment and install what you need:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the App

Start the application with:
```bash
streamlit run app/app.py
```

The app will open in your web browser.

## How to Use

1. **Generate Data**: Create test marketing data
2. **Transform Data**: Apply adjustments to model real-world marketing effects
3. **Train Models**: Build statistical models to understand marketing impact
4. **Analyze Results**: See how each marketing channel contributes to sales
5. **Optimize Budget**: Find the best way to allocate your marketing spend

## What You Need

- Python 3.7 or newer
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## Project Structure

```
QuantumMMM/
│
├── app/                    # Main application
├── mmm/                    # Core modeling components
├── data/                   # Data storage
├── notebooks/              # Jupyter notebooks
├── tests/                  # Test files
│
├── README.md               # This file
├── requirements.txt        # Dependencies
└── setup.py                # Installation settings
```

## About This Project

This project was created as a demonstration of Marketing Mix Modeling concepts and techniques. It's designed for educational purposes and to showcase marketing analytics skills.

Created for a fifty-five internship application.
