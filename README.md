Impulse Buying Prediction

1. Files in the folder: 
- `train_models.py` – runs the machine-learning experiments
- `Untitled form.csv` – Google Forms responses
- `social_media_buying.csv` – social media buying data
- `Cleaned dataset.xlsx` – psychological drivers data
- `requirements.txt` – Python libraries needed
- `results` – model results, graphs and saved models

 To run the project:
1. Open this folder in VS Code.
2. Open the terminal.
3. Install the libraries:

```bash
pip install -r requirements.txt
```

4. Run the program:

```bash
python train_models.py
```

5. Check the `results` folder.

The program compares Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and a Combined Voting Model. It evaluates each dataset separately and saves the comparison tables, graphs, confusion matrices and selected models.

