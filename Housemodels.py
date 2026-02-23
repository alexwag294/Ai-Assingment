import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class HousePricePredictor:

    def __init__(self, root):
        self.root = root
        self.root.title("🏠 Advanced House Price Predictor")
        self.root.geometry("900x950")
        self.root.configure(bg="#1254a9")

        self.model = None
        self.best_model_name = None
        self.df = None
        self.locations = []

        # Store model performance
        self.lr_r2 = None
        self.rf_r2 = None
        self.lr_cv = None
        self.rf_cv = None

        self.create_ui()

    # ================= UI =================
    def create_ui(self):

        title = tk.Label(
            self.root,
            text="🏠 ADVANCED HOUSE PRICE PREDICTOR",
            font=("Arial", 20, "bold"),
            bg="#1e1e2f",
            fg="white"
        )
        title.pack(pady=10)

        self.result_label = tk.Label(
            self.root,
            text="💰 Predicted price will appear here",
            font=("Arial", 14, "bold"),
            bg="#1e1e2f",
            fg="#00ffcc"
        )
        self.result_label.pack(pady=10)

        form = tk.Frame(self.root, bg="#4c89d8")
        form.pack(pady=10, padx=40, fill="both", expand=True)

        labels = [
            "Area (sq ft)", "Bedrooms", "Bathrooms", "Age",
            "Garage Cars", "Garage Area", "Basement Area",
            "Overall Quality (1-10)", "Lot Area", "Year Built"
        ]

        self.entries = {}

        for i, label in enumerate(labels):
            tk.Label(form, text=label, bg="#1254a9",
                     fg="white", font=("Arial", 11)
                     ).grid(row=i, column=0, pady=8, sticky="e")

            entry = tk.Entry(form, font=("Arial", 11))
            entry.grid(row=i, column=1, pady=8, padx=10)

            self.entries[label] = entry

        tk.Label(form, text="Neighborhood", bg="#874ce0",
                 fg="white", font=("Arial", 11)
                 ).grid(row=len(labels), column=0, pady=8, sticky="e")

        self.location_var = tk.StringVar()
        self.location_box = ttk.Combobox(
            form,
            textvariable=self.location_var,
            font=("Arial", 11),
            state="disabled"
        )
        self.location_box.grid(row=len(labels), column=1, pady=8, padx=10)

        tk.Button(self.root,
                  text="📂 Load CSV & Train Models",
                  font=("Arial", 12, "bold"),
                  bg="#4CAF50",
                  fg="white",
                  command=self.load_and_train).pack(pady=10)

        tk.Button(self.root,
                  text="🔮 Predict Price",
                  font=("Arial", 12, "bold"),
                  bg="#ff9800",
                  fg="white",
                  command=self.predict_price).pack(pady=10)

        tk.Button(self.root,
                  text="📊 Show Graphs",
                  font=("Arial", 12, "bold"),
                  bg="#2196F3",
                  fg="white",
                  command=self.show_graphs).pack(pady=10)

    # ================= TRAIN =================
    def load_and_train(self):
        try:
            self.df = pd.read_csv(
                r"C:\Users\Acer\Downloads\house-prices-advanced-regression-techniques\train.csv"
            )

            self.df["Age"] = self.df["YrSold"] - self.df["YearBuilt"]

            numeric_features = [
                "GrLivArea", "BedroomAbvGr", "FullBath", "Age",
                "GarageCars", "GarageArea", "TotalBsmtSF",
                "OverallQual", "LotArea", "YearBuilt"
            ]
            categorical_features = ["Neighborhood"]

            X = self.df[numeric_features + categorical_features]
            y = self.df["SalePrice"]

            self.locations = sorted(self.df["Neighborhood"].unique())
            self.location_box.config(values=self.locations, state="readonly")
            self.location_var.set(self.locations[0])

            numeric_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ])

            lr_model = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression())
            ])

            rf_model = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=200,
                    random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            lr_model.fit(X_train, y_train)
            rf_model.fit(X_train, y_train)

            self.lr_r2 = lr_model.score(X_test, y_test)
            self.rf_r2 = rf_model.score(X_test, y_test)

            self.lr_cv = cross_val_score(lr_model, X, y, cv=5).mean()
            self.rf_cv = cross_val_score(rf_model, X, y, cv=5).mean()

            if self.rf_r2 > self.lr_r2:
                self.model = rf_model
                self.best_model_name = "Random Forest"
                best_r2 = self.rf_r2
            else:
                self.model = lr_model
                self.best_model_name = "Linear Regression"
                best_r2 = self.lr_r2

            messagebox.showinfo(
                "Training Complete",
                f"Linear Regression R²: {self.lr_r2:.3f}\n"
                f"Random Forest R²: {self.rf_r2:.3f}\n\n"
                f"LR Cross-Val: {self.lr_cv:.3f}\n"
                f"RF Cross-Val: {self.rf_cv:.3f}\n\n"
                f"Best Model: {self.best_model_name}\n"
                f"Best R²: {best_r2:.3f}"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ================= PREDICT =================
    def predict_price(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Train the model first!")
            return

        try:
            data = {
                "GrLivArea": [float(self.entries["Area (sq ft)"].get())],
                "BedroomAbvGr": [int(self.entries["Bedrooms"].get())],
                "FullBath": [int(self.entries["Bathrooms"].get())],
                "Age": [int(self.entries["Age"].get())],
                "GarageCars": [int(self.entries["Garage Cars"].get())],
                "GarageArea": [float(self.entries["Garage Area"].get())],
                "TotalBsmtSF": [float(self.entries["Basement Area"].get())],
                "OverallQual": [int(self.entries["Overall Quality (1-10)"].get())],
                "LotArea": [float(self.entries["Lot Area"].get())],
                "YearBuilt": [int(self.entries["Year Built"].get())],
                "Neighborhood": [self.location_var.get()]
            }

            input_df = pd.DataFrame(data)
            prediction = self.model.predict(input_df)[0]

            self.result_label.config(
                text=f"💰 Estimated Price ({self.best_model_name}): ${prediction:,.2f}"
            )

        except:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")

    # ================= SHOW GRAPHS =================
    def show_graphs(self):
        if self.df is None or self.model is None:
            messagebox.showwarning("Warning", "Load and train the model first!")
            return

        numeric_features = [
            "GrLivArea", "BedroomAbvGr", "FullBath", "Age",
            "GarageCars", "GarageArea", "TotalBsmtSF",
            "OverallQual", "LotArea", "YearBuilt"
        ]
        categorical_features = ["Neighborhood"]

        X = self.df[numeric_features + categorical_features]
        y_true = self.df["SalePrice"]
        y_pred = self.model.predict(X)
        residuals = y_true - y_pred

        # Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(y_true, bins=30, color="lightyellow", edgecolor="black")
        plt.title("Distribution of Sale Prices")
        plt.show()

        # Predicted vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, color="blue")
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 color="darkblue", linestyle="--")
        plt.title("Predicted vs Actual Prices")
        plt.show()

        # Residual Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, residuals, alpha=0.6, color="lightgreen")
        plt.axhline(0, color="green", linestyle="--")
        plt.title("Actual Price vs Prediction Error")
        plt.show()

        # Living Area vs Residuals
        plt.figure(figsize=(8, 6))
        plt.scatter(self.df["GrLivArea"], residuals,
                    alpha=0.6, color="violet")
        plt.axhline(0, color="purple", linestyle="--")
        plt.title("Living Area vs Prediction Error")
        plt.show()

        # Feature Importance
        if self.best_model_name == "Random Forest":

            feature_names = numeric_features + list(
                self.model.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names_out(categorical_features)
            )

            importances = self.model.named_steps["regressor"].feature_importances_
            indices = np.argsort(importances)[-15:]

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)),
                     importances[indices],
                     color="orange")
            plt.yticks(range(len(indices)),
                       [feature_names[i] for i in indices])
            plt.title("Top 15 Feature Importances (Random Forest)")
            plt.show()

        # Model Comparison Chart
        models = ["Linear Regression", "Random Forest"]
        r2_scores = [self.lr_r2, self.rf_r2]
        cv_scores = [self.lr_cv, self.rf_cv]

        x = np.arange(len(models))
        width = 0.35

        plt.figure(figsize=(8, 6))
        plt.bar(x - width/2, r2_scores, width, label="R²", color="skyblue")
        plt.bar(x + width/2, cv_scores, width, label="Cross-Val", color="lightcoral")
        plt.xticks(x, models)
        plt.title("Model Performance Comparison")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = HousePricePredictor(root)
    root.mainloop()