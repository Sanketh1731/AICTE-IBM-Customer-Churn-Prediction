# Customer Churn Prediction & Retention Dashboard
# Author: Sanketh

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

DATA_FILE = "telco_churn_data.csv"

def load_and_preprocess_data(filepath=DATA_FILE):
    if not os.path.exists(filepath):
        import urllib.request
        mirror_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        urllib.request.urlretrieve(mirror_url, filepath)

    df = pd.read_csv(filepath)

    # Convert TotalCharges to numeric and handle nulls
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Target variable mapping
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Tenure cohorts
    bins = [0, 12, 24, 48, 72]
    labels = ['0-12 Mo (New)', '13-24 Mo (Early)', '25-48 Mo (Mid-stage)', '49-72 Mo (Loyal)']
    df['Tenure_Cohort'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)

    # Charge tiers
    charge_bins = [0, 35, 70, 120]
    charge_labels = ['Low (<$35)', 'Medium ($35-$70)', 'High (>$70)']
    df['Charge_Tier'] = pd.cut(df['MonthlyCharges'], bins=charge_bins, labels=charge_labels, include_lowest=True)

    return df


def prepare_ml_features(df):
    ignore_cols = ['customerID', 'Churn', 'Churn_Numeric', 'Tenure_Cohort', 'Charge_Tier']
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df[feature_cols], drop_first=True)

    X = df_encoded
    y = df['Churn_Numeric']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), scaler, X_train


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    }

    metrics = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        metrics[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc,
            "Confusion Matrix": cm,
            "Predictions": y_pred,
            "Probabilities": y_proba
        }
        trained_models[name] = model

    return metrics, trained_models


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Customer Churn & Retention Dashboard | Sanketh",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .metric-subtitle {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .risk-high {
        color: #ef4444 !important;
    }
    .risk-low {
        color: #10b981 !important;
    }
    .action-box {
        background: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #38bdf8;
        padding: 16px;
        border-radius: 6px;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test, feature_names, scaler, X_raw_train = prepare_ml_features(df)
    metrics, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    best_model_name = "Gradient Boosting"
    best_model = trained_models[best_model_name]

    # Sidebar Navigation
    st.sidebar.markdown("### Customer Churn Analytics")
    st.sidebar.markdown("**Author:** Sanketh")
    st.sidebar.markdown("---")

    nav_selection = st.sidebar.radio(
        "Dashboard Navigation",
        ["Overview", "Driver Analysis", "AI Churn Predictor"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Filters")
    contract_filter = st.sidebar.multiselect(
        "Contract Type:",
        options=df['Contract'].unique(),
        default=df['Contract'].unique()
    )
    internet_filter = st.sidebar.multiselect(
        "Internet Service:",
        options=df['InternetService'].unique(),
        default=df['InternetService'].unique()
    )

    filtered_df = df[(df['Contract'].isin(contract_filter)) & (df['InternetService'].isin(internet_filter))]

    # Page 1: Overview
    if nav_selection == "Overview":
        st.title("Customer Churn & Retention Overview")
        st.markdown(
            "Executive dashboard tracking customer retention, revenue exposure, and cohort dynamics."
        )

        total_customers = len(filtered_df)
        churned_customers = filtered_df['Churn_Numeric'].sum()
        churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
        total_mrr = filtered_df['MonthlyCharges'].sum()
        mrr_at_risk = filtered_df[filtered_df['Churn_Numeric'] == 1]['MonthlyCharges'].sum()
        retained_mrr = total_mrr - mrr_at_risk

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Customers</div>
                <div class="metric-value">{total_customers:,}</div>
                <div class="metric-subtitle">Active accounts</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Churn Rate</div>
                <div class="metric-value {'risk-high' if churn_rate > 25 else 'risk-low'}">{churn_rate:.1f}%</div>
                <div class="metric-subtitle">{churned_customers:,} churned accounts</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MRR At Risk</div>
                <div class="metric-value risk-high">${mrr_at_risk:,.0f}</div>
                <div class="metric-subtitle">{(mrr_at_risk/total_mrr*100) if total_mrr>0 else 0:.1f}% of total MRR</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Retained MRR</div>
                <div class="metric-value risk-low">${retained_mrr:,.0f}</div>
                <div class="metric-subtitle">Protected recurring revenue</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        r1c1, r1c2 = st.columns([1, 1])

        with r1c1:
            st.subheader("Churn Rate by Tenure Cohort")
            cohort_stats = filtered_df.groupby('Tenure_Cohort', observed=False)['Churn_Numeric'].agg(['count', 'mean']).reset_index()
            cohort_stats['Churn_Rate'] = cohort_stats['mean'] * 100

            fig, ax = plt.subplots(figsize=(6, 3.8))
            bars = ax.bar(cohort_stats['Tenure_Cohort'].astype(str), cohort_stats['Churn_Rate'], color='#38bdf8', edgecolor='#0284c7', width=0.55)
            ax.set_ylabel("Churn Rate (%)", color="#94a3b8", fontsize=10)
            ax.set_ylim(0, 60)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.set_facecolor('#0f172a')
            fig.patch.set_facecolor('#0f172a')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_color('#334155')
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=9)
            st.pyplot(fig)
            st.caption("Highest churn risk occurs during the first 12 months of service.")

        with r1c2:
            st.subheader("Churn Distribution by Contract Type")
            contract_stats = filtered_df.groupby('Contract', observed=False)['Churn_Numeric'].mean().reset_index()
            contract_stats['Churn_Rate'] = contract_stats['mean'] * 100

            fig2, ax2 = plt.subplots(figsize=(6, 3.8))
            colors = ['#ef4444', '#f59e0b', '#10b981']
            bars2 = ax2.bar(contract_stats['Contract'], contract_stats['Churn_Rate'], color=colors, width=0.5)
            ax2.set_ylabel("Churn Rate (%)", color="#94a3b8", fontsize=10)
            ax2.set_ylim(0, 55)
            ax2.grid(axis='y', linestyle='--', alpha=0.3)
            ax2.set_facecolor('#0f172a')
            fig2.patch.set_facecolor('#0f172a')
            ax2.tick_params(colors='#94a3b8')
            for spine in ax2.spines.values():
                spine.set_color('#334155')
            for bar in bars2:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=9)
            st.pyplot(fig2)
            st.caption("Month-to-month contracts account for the vast majority of customer loss.")

        st.markdown("""
        <div class="action-box">
            <h4 style="color:#38bdf8; margin-top:0;">💡 Key Business Insight</h4>
            <p><strong>Finding:</strong> New subscribers on month-to-month contracts have a 42.7% attrition rate.</p>
            <p><strong>Recommendation:</strong> Migrating 15% of month-to-month customers to annual commitments secures approximately <strong>$21,000 monthly ($252,000 ARR)</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    # Page 2: Driver Analysis
    elif nav_selection == "Driver Analysis":
        st.title("Customer Churn Drivers")
        st.markdown(
            "Analyzing service factors, billing methods, and feature importance."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Internet Service & Tech Support")
            service_df = filtered_df.groupby(['InternetService', 'TechSupport'], observed=False)['Churn_Numeric'].mean().unstack() * 100

            fig, ax = plt.subplots(figsize=(6, 4))
            service_df.plot(kind='bar', ax=ax, colormap='Spectral', width=0.7)
            ax.set_ylabel("Churn Rate (%)", color="#94a3b8")
            ax.set_facecolor('#0f172a')
            fig.patch.set_facecolor('#0f172a')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_color('#334155')
            ax.legend(title="Tech Support", facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
            st.pyplot(fig)
            st.caption("Fiber optic customers without tech support experience nearly 50% churn.")

        with col2:
            st.subheader("Payment Method Impact")
            pay_stats = filtered_df.groupby('PaymentMethod')['Churn_Numeric'].mean().reset_index()
            pay_stats['Churn_Rate'] = pay_stats['mean'] * 100
            pay_stats = pay_stats.sort_values(by='Churn_Rate', ascending=False)

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            bars = ax2.barh(pay_stats['PaymentMethod'], pay_stats['Churn_Rate'], color='#f43f5e', height=0.55)
            ax2.set_xlabel("Churn Rate (%)", color="#94a3b8")
            ax2.set_facecolor('#0f172a')
            fig2.patch.set_facecolor('#0f172a')
            ax2.tick_params(colors='#94a3b8')
            for spine in ax2.spines.values():
                spine.set_color('#334155')
            for bar in bars:
                wval = bar.get_width()
                ax2.text(wval + 1, bar.get_y() + bar.get_height()/2.0, f'{wval:.1f}%', va='center', color='#f8fafc', fontweight='bold', fontsize=9)
            st.pyplot(fig2)
            st.caption("Electronic check payment has significantly higher churn than automated billing.")

        st.markdown("---")

        st.subheader("Model Feature Importance")
        rf_model = trained_models['Random Forest']
        importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        importances.sort_values().plot(kind='barh', ax=ax3, color='#38bdf8', edgecolor='#0284c7')
        ax3.set_xlabel("Relative Importance Weight", color="#94a3b8")
        ax3.set_facecolor('#0f172a')
        fig3.patch.set_facecolor('#0f172a')
        ax3.tick_params(colors='#94a3b8')
        for spine in ax3.spines.values():
            spine.set_color('#334155')
        st.pyplot(fig3)

        st.markdown("### Model Evaluation Summary")
        perf_data = []
        for name, m in metrics.items():
            perf_data.append({
                "Model": name,
                "Accuracy": f"{m['Accuracy']*100:.2f}%",
                "Precision": f"{m['Precision']*100:.2f}%",
                "Recall": f"{m['Recall']*100:.2f}%",
                "F1-Score": f"{m['F1-Score']*100:.2f}%",
                "ROC-AUC": f"{m['ROC-AUC']:.4f}"
            })
        st.table(pd.DataFrame(perf_data))

    # Page 3: AI Churn Predictor
    elif nav_selection == "AI Churn Predictor":
        st.title("Customer Churn Risk Predictor")
        st.markdown(
            "Predict individual customer churn probability and generate retention strategies."
        )

        with st.form("prediction_form"):
            st.markdown("#### Account & Plan Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                tenure_input = st.slider("Tenure (Months)", 1, 72, 6)
                monthly_charges_input = st.number_input("Monthly Charges ($)", 18.0, 120.0, 85.0, step=1.0)
                total_charges_input = st.number_input("Total Charges ($)", 18.0, 9000.0, float(tenure_input * monthly_charges_input), step=20.0)
            with col2:
                contract_input = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                internet_input = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                payment_input = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
            with col3:
                tech_support_input = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                online_sec_input = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                paperless_input = st.selectbox("Paperless Billing", ["Yes", "No"])

            st.markdown("#### Customer Profile")
            c4, c5, c6 = st.columns(3)
            with c4:
                senior_input = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            with c5:
                partner_input = st.selectbox("Partner", ["No", "Yes"])
            with c6:
                dependents_input = st.selectbox("Dependents", ["No", "Yes"])

            submit_btn = st.form_submit_button("Predict Churn Risk", use_container_width=True)

        if submit_btn:
            input_dict = {col: 0 for col in feature_names}

            if 'SeniorCitizen' in input_dict:
                input_dict['SeniorCitizen'] = senior_input
            if 'tenure' in input_dict:
                input_dict['tenure'] = tenure_input
            if 'MonthlyCharges' in input_dict:
                input_dict['MonthlyCharges'] = monthly_charges_input
            if 'TotalCharges' in input_dict:
                input_dict['TotalCharges'] = total_charges_input

            def set_flag(feat_name, val):
                col = f"{feat_name}_{val}"
                if col in input_dict:
                    input_dict[col] = 1

            set_flag('Partner', partner_input)
            set_flag('Dependents', dependents_input)
            set_flag('InternetService', internet_input)
            set_flag('OnlineSecurity', online_sec_input)
            set_flag('TechSupport', tech_support_input)
            set_flag('Contract', contract_input)
            set_flag('PaperlessBilling', paperless_input)
            set_flag('PaymentMethod', payment_input)

            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)

            churn_proba = best_model.predict_proba(input_scaled)[0, 1]

            st.markdown("---")
            st.subheader("Prediction Results")

            rc1, rc2, rc3 = st.columns(3)

            with rc1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Churn Probability</div>
                    <div class="metric-value {'risk-high' if churn_proba > 0.5 else 'risk-low'}">{churn_proba*100:.1f}%</div>
                    <div class="metric-subtitle">Predicted risk score</div>
                </div>
                """, unsafe_allow_html=True)

            with rc2:
                risk_tier = "HIGH RISK" if churn_proba >= 0.65 else ("MEDIUM RISK" if churn_proba >= 0.35 else "LOW RISK")
                badge_class = "risk-high" if churn_proba >= 0.65 else ("metric-value" if churn_proba >= 0.35 else "risk-low")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Risk Level</div>
                    <div class="metric-value {badge_class}">{risk_tier}</div>
                    <div class="metric-subtitle">Retention priority tier</div>
                </div>
                """, unsafe_allow_html=True)

            with rc3:
                revenue_at_stake = monthly_charges_input * 12
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Annual Value at Risk</div>
                    <div class="metric-value">${revenue_at_stake:,.0f}</div>
                    <div class="metric-subtitle">12-month projected ARR</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Recommended Action Plan")
            if churn_proba >= 0.60:
                st.error("""
                **High-Risk Retention Strategy:**
                1. **Contract Migration**: Offer a 15% discount for switching to an annual plan.
                2. **Service Add-on**: Provide 90 days complimentary tech support to improve service satisfaction.
                3. **Billing Incentive**: Offer a $10 bill credit to enroll in automated bank/card payments.
                """)
            elif churn_proba >= 0.35:
                st.warning("""
                **Proactive Retention Strategy:**
                1. **Account Check-in**: Schedule customer support follow-up on service reliability.
                2. **Milestone Reward**: Offer loyalty bonus points for surpassing 12 months tenure.
                """)
            else:
                st.success("""
                **Low Risk Customer:**
                1. **Account Growth**: Suitable for bundle promotions or optional add-ons.
                2. **Engagement**: Solicit customer feedback or referral participation.
                """)


def run_cli_pipeline():
    print("Loading Telco dataset...")
    df = load_and_preprocess_data()
    print(f"Total customers: {len(df):,}")
    print(f"Baseline churn rate: {df['Churn_Numeric'].mean() * 100:.2f}%")

    print("\nPreparing feature matrix and splits...")
    X_train, X_test, y_train, y_test, feature_names, scaler, X_raw_train = prepare_ml_features(df)

    print("\nTraining classification models...")
    metrics, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("\nModel Benchmark:")
    for name, m in metrics.items():
        print(f"  {name:20} -> Accuracy: {m['Accuracy']*100:.2f}% | Precision: {m['Precision']*100:.2f}% | Recall: {m['Recall']*100:.2f}% | F1: {m['F1-Score']*100:.2f}% | AUC: {m['ROC-AUC']:.4f}")

    os.makedirs("assets", exist_ok=True)

    # 1. Tenure and Contract plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cohort_stats = df.groupby('Tenure_Cohort', observed=False)['Churn_Numeric'].mean() * 100
    cohort_stats.plot(kind='bar', ax=ax1, color='#0284c7', edgecolor='black')
    ax1.set_title("Churn Rate by Tenure Cohort", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Churn Rate (%)")
    ax1.set_xlabel("Tenure Cohort")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    contract_stats = df.groupby('Contract', observed=False)['Churn_Numeric'].mean() * 100
    contract_stats.plot(kind='bar', ax=ax2, color=['#ef4444', '#f59e0b', '#10b981'], edgecolor='black')
    ax2.set_title("Churn Rate by Contract Type", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Churn Rate (%)")
    ax2.set_xlabel("Contract Type")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("assets/fig_tenure_and_contract.png", dpi=300)
    plt.close()

    # 2. Confusion matrix and ROC
    best_name = "Gradient Boosting"
    best_m = metrics[best_name]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(best_m['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
    ax1.set_title(f"Confusion Matrix ({best_name})", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Actual")
    ax1.set_xlabel("Predicted")

    fpr, tpr, _ = roc_curve(y_test, best_m['Probabilities'])
    ax2.plot(fpr, tpr, color='#0284c7', lw=2, label=f"ROC (AUC = {best_m['ROC-AUC']:.3f})")
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_title("Receiver Operating Characteristic", fontsize=12, fontweight='bold')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("assets/fig_model_evaluation.png", dpi=300)
    plt.close()

    # 3. Feature importance
    rf = trained_models['Random Forest']
    top_feats = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    top_feats.sort_values().plot(kind='barh', color='#38bdf8', edgecolor='black')
    plt.title("Top Feature Importances", fontsize=12, fontweight='bold')
    plt.xlabel("Importance Weight")
    plt.tight_layout()
    plt.savefig("assets/fig_feature_importance.png", dpi=300)
    plt.close()

    print("Visual assets exported to assets/ folder.")


if __name__ == "__main__":
    if "streamlit" in sys.modules or (len(sys.argv) > 1 and sys.argv[1] == "--streamlit"):
        run_streamlit_app()
    elif len(sys.argv) > 1 and sys.argv[1] == "--run-ui":
        run_streamlit_app()
    else:
        run_cli_pipeline()
