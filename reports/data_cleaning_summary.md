## Project Objectives

- Clean and improve the Titanic dataset from Kaggle using Pandas in Google Colab  
- Fix common data problems like missing values, duplicate rows, and inconsistent formats  
- Create new features (for example, FamilySize) and handle unusual/outlier values  
- Prepare the dataset so it can be used for data analysis, visualization, or AI models

## Key Skills Demonstrated

- **Data Cleaning:** Fill missing values (using median or mode), remove unnecessary columns, and handle duplicate rows  
- **Data Standardization:** Make data consistent (for example, lowercase text and rounded numbers)  
- **Feature Engineering:** Create new useful features like `FamilySize` from `SibSp` and `Parch`  
- **Outlier Management:** Detect and handle unusual values using the IQR method  
- **Documentation:** Explain each step clearly so anyone can follow your work  
- **Reproducibility:** Organize code so the cleaning process can be repeated easily

## Dataset Information

- **Source:** Titanic dataset from GitHub (Data Science Dojo repository)  
- **Original Size:** 891 rows × 12 columns  
- **Key Columns:** PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked  

**Data Issues:**  
- **Age:** 20% missing values  
- **Cabin:** 77% missing values  
- **Embarked:** A few missing values  
- **Other:** Possible duplicate rows and inconsistent formats  

---

## Code Walkthrough

**Notebook:** `titanic_data_cleaning.ipynb` – Main Workflow  

### Task 1 – Handle Missing Data
```python
# Fill missing Age values with median (good for data with outliers)
# Remove Cabin column (too many missing values)
# Fill missing Embarked values with the most common value (mode)
# Result: DataFrame has no missing Age or Embarked values

Task 2 – Remove Duplicates

# Check for exact duplicate rows
# Remove duplicates, keep the first occurrence
# Result: Usually 0 duplicates in Titanic dataset

Task 3 – Standardize Formats

# Convert 'Sex' column to lowercase ('male' or 'female')
# Round 'Fare' values to 2 decimals for readability
# Ensures all data is consistent and ready for analysis

Task 4 – Create Feature and Handle Outliers

# Create new feature: FamilySize = SibSp + Parch + 1 (including passenger)
# Detect Fare outliers using IQR method:
#   Q1 = 7.91
#   Q3 = 31.00
#   IQR = 23.09
#   Lower bound = Q1 - 1.5*IQR = -26.72 (set to 0 for practicality)
#   Upper bound = Q3 + 1.5*IQR = 65.63
# Cap outliers within these bounds using .clip() method

## Best Practices & Improvements

### Strengths
- Tasks are clearly separated with good documentation  
- Used `df.copy()` to keep the original data safe  
- Checked the data at each step to make sure everything is correct  
- Used good imputation methods (median for Age, mode for Embarked)  

### Suggested Improvements

**Code Quality**
1. Add error handling with try-except for file loading and other operations  
2. Fix column name typos (e.g., `$ibSp` → `SibSp`)  
3. Make reusable functions for each cleaning step  
4. Use a configuration file to store constants like URLs, column names, and bounds  

**Data Processing**
1. Improve Age imputation by using groups (like Pclass/Sex) instead of overall median  
2. Extract deck info from Cabin before dropping the column (use the first letter)  
3. Extract titles from Name (Mr., Mrs., Miss, etc.) as a new feature  
4. Detect outliers more carefully by combining IQR with domain knowledge  

**Documentation**
1. Add a data dictionary explaining each column and what was changed  
2. Include before/after charts for Age and Fare distributions  
3. Keep a log of all assumptions made during cleaning

## Project Structure

1. **Modular Design:** Split the project into separate parts for data loading, cleaning, and feature creation  
2. **Testing:** Add small tests to check that each cleaning step works correctly  
3. **Pipeline:** Make a workflow (like sklearn pipeline) so the cleaning can be repeated easily  

---

# Titanic Data Cleaning & Feature Engineering Project

## Project Overview
This project shows how to clean and improve the Titanic dataset from Kaggle. It fixes common data problems like missing values, duplicate rows, inconsistent formats, and outliers. It also creates new features that make the data ready for analysis or modeling.
  
## Objectives
- **Clean raw data:** Fix missing values, remove duplicates, and standardize formats  
- **Create features:** Make new columns (like FamilySize) to help with analysis or prediction  
- **Handle outliers:** Find and treat extreme values using simple methods  
- **Prepare for analysis:** Make the dataset ready for charts, visualization, or machine learning  
- **Document process:** Write clear, step-by-step instructions so anyone can repeat the work

##  Skills Demonstrated
- **Data Cleaning**: Imputation (median/mode), duplicate removal, format standardization
- **Feature Engineering**: Creating meaningful derived features (FamilySize)
- **Outlier Management**: Statistical detection (IQR method) and treatment (capping)
- **Data Validation**: Systematic checking at each processing stage
- **Documentation**: Clear, commented code with step-by-step explanations
- **Pandas Proficiency**: Advanced DataFrame manipulation and transformation

## Tools & Technologies
- **Python 3.12+**
- **Pandas**: Data manipulation and cleaning
- **NumPy**: Numerical operations
- **Google Colab**: Cloud-based notebook environment
- **Git/GitHub**: Version control and project hosting

##  Project Structure
```

titanic-data-cleaning/
├──notebooks/
│├── titanic_data_cleaning.ipynb     # Main cleaning workflow
│└── feature_engineering.ipynb       # Advanced feature creation
├──data/
│├── raw/titanic.csv                 # Original dataset
│└── processed/titanic_clean.csv     # Cleaned dataset
├──images/                             # Visualizations and charts
├──reports/
│└── data_cleaning_summary.md        # Detailed process documentation
├──requirements.txt                    # Dependencies
└──README.md                           # This file

```

##  Data Cleaning Workflow

### Step 1: Handle Missing Data
| Column | Issue | Treatment | Justification |
|--------|-------|-----------|---------------|
| Age | 20% missing | Imputed with median (28.0) | Robust to outliers, preserves distribution |
| Cabin | 77% missing | Column dropped | Too many missing for reliable imputation |
| Embarked | 2 missing | Filled with mode ('S') | Categorical variable, mode is appropriate |

**Before**: 177 missing Age, 687 missing Cabin, 2 missing Embarked  
**After**: 0 missing values in retained columns

### Step 2: Remove Duplicates
- Checked for exact duplicate rows across all columns
- Result: 0 duplicates found in Titanic dataset
- Process maintained for general data integrity

### Step 3: Standardize Formats
- **Sex**: Converted to lowercase ('male', 'female')
- **Fare**: Rounded to 2 decimal places (currency format)

### Step 4: Feature Engineering & Outlier Management
1. **Created FamilySize**:
   - Formula: `SibSp + Parch + 1`
   - Range: 1 (alone) to 11 (large family)
   - Distribution: 537 passengers traveling alone

2. **Handled Fare Outliers**:
   - Method: Interquartile Range (IQR) with 1.5× multiplier
   - Bounds: Lower = -26.72 (capped at 0), Upper = 65.63
   - Outliers identified: 0 (dataset already within reasonable range)
   - Cap applied to prevent future outlier issues

##  Getting Started

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/titanic_data_cleaning.ipynb`
3. Run cells sequentially (Runtime → Run All)

### Option 2: Local Installation
```bash
# Clone repository
git clone https://github.com/yourusername/titanic-data-cleaning.git
cd titanic-data-cleaning

# Install dependencies
## Dependencies

To run the notebooks, install the following packages:

```bash
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook notebooks/titanic_data_cleaning.ipynb
