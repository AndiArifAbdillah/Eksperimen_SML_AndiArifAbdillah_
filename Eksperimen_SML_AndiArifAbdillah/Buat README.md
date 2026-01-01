# Eksperimen Machine Learning System
## Wine Quality Prediction

**Author:** Andi Arif Abdillah  
**Email:** andiarifabc@gmail.com  
**Date:** December 2025

## Dataset
- **Source:** UCI Machine Learning Repository
- **Name:** Wine Quality Dataset (Red Wine)
- **Features:** 11 chemical properties
- **Target:** Quality score (converted to binary classification)

## Project Structure
```
Eksperimen_SML_AndiArifAbdillah/
├── .github/
│   └── workflows/
│       └── preprocessing.yml
├── data_raw/
│   └── wine_quality_raw.csv
├── preprocessing/
│   ├── Eksperimen_AndiArifAbdillah.ipynb
│   ├── automate_AndiArifAbdillah.py
│   └── data_preprocessing/
│       ├── wine_train_processed.csv
│       └── wine_test_processed.csv
└── README.md
```

## How to Run

### Manual Preprocessing (Notebook)
1. Open `preprocessing/Eksperimen_AndiArifAbdillah.ipynb`
2. Run all cells sequentially
3. Check output in `preprocessing/data_preprocessing/`

### Automated Preprocessing (Script)
```bash
cd preprocessing
python automate_AndiArifAbdillah.py
```

### Automated via GitHub Actions
Push any changes to main branch, and the workflow will automatically run preprocessing.

## Preprocessing Steps
1. Load raw data
2. Handle missing values
3. Remove duplicates
4. Remove outliers using IQR method
5. Feature engineering (binary classification)
6. Train-test split (80-20)
7. Feature scaling (StandardScaler)
8. Save processed data

## Results
- Training samples: ~1000
- Test samples: ~200
- Features: 11 numerical features
- Target: Binary (Good wine: 1, Bad wine: 0)