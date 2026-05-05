# to run, paste this at the end of the notebook
import joblib
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Rebuild a cleaned raw-data frame so the dashboard can accept the original 13 clinical inputs.
raw_heart_df = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1).copy()
raw_heart_df.replace('?', np.nan, inplace=True)
raw_heart_df.dropna(inplace=True)
raw_heart_df.rename(columns={'num': 'target'}, inplace=True)
raw_heart_df['target'] = raw_heart_df['target'].apply(lambda value: 1 if value > 0 else 0)

# Save the deployed model artifact from the notebook.
model_save_path = Path('/home/fatima/ASemester6_S26/DM/asst4/notebooks/heart_disease_rf_best_model.joblib')
joblib.dump(rf_best_model, model_save_path)
print(f'Saved model to: {model_save_path}')
print('This notebook cell provides the E-section dashboard directly inside the notebook.')

feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
feature_hints = {
    'age': '29-77 years',
    'sex': '0=female, 1=male',
    'cp': 'categorical code',
    'trestbps': 'typ. 90-200 mmHg',
    'chol': 'typ. 120-600 mg/dl',
    'fbs': '0/1 indicator',
    'restecg': 'categorical code',
    'thalach': 'typ. 70-210 bpm',
    'exang': '0/1 indicator',
    'oldpeak': 'typ. 0-6.5',
    'slope': 'categorical code',
    'ca': '0-3 vessels',
    'thal': 'categorical code',
}

sample_patient = raw_heart_df.iloc[0]

def make_dropdown_options(series):
    values = list(pd.Series(series).dropna().unique())
    values = sorted(values)
    return [(str(value), value) for value in values]

widget_specs = {
    'age': widgets.BoundedFloatText(value=float(sample_patient['age']), min=float(raw_heart_df['age'].min()), max=float(raw_heart_df['age'].max()), step=1.0, description=f'Age ({feature_hints["age"]})', layout=widgets.Layout(width='320px')),
    'sex': widgets.Dropdown(options=[('0', 0), ('1', 1)], value=int(sample_patient['sex']), description=f'Sex ({feature_hints["sex"]})', layout=widgets.Layout(width='320px')),
    'cp': widgets.Dropdown(options=make_dropdown_options(raw_heart_df['cp']), value=sample_patient['cp'], description=f'CP ({feature_hints["cp"]})', layout=widgets.Layout(width='320px')),
    'trestbps': widgets.BoundedFloatText(value=float(sample_patient['trestbps']), min=float(raw_heart_df['trestbps'].min()), max=float(raw_heart_df['trestbps'].max()), step=1.0, description=f'Trestbps ({feature_hints["trestbps"]})', layout=widgets.Layout(width='320px')),
    'chol': widgets.BoundedFloatText(value=float(sample_patient['chol']), min=float(raw_heart_df['chol'].min()), max=float(raw_heart_df['chol'].max()), step=1.0, description=f'Chol ({feature_hints["chol"]})', layout=widgets.Layout(width='320px')),
    'fbs': widgets.Dropdown(options=[('0', 0), ('1', 1)], value=int(sample_patient['fbs']), description=f'FBS ({feature_hints["fbs"]})', layout=widgets.Layout(width='320px')),
    'restecg': widgets.Dropdown(options=make_dropdown_options(raw_heart_df['restecg']), value=sample_patient['restecg'], description=f'Restecg ({feature_hints["restecg"]})', layout=widgets.Layout(width='320px')),
    'thalach': widgets.BoundedFloatText(value=float(sample_patient['thalach']), min=float(raw_heart_df['thalach'].min()), max=float(raw_heart_df['thalach'].max()), step=1.0, description=f'Thalach ({feature_hints["thalach"]})', layout=widgets.Layout(width='320px')),
    'exang': widgets.Dropdown(options=[('0', 0), ('1', 1)], value=int(sample_patient['exang']), description=f'Exang ({feature_hints["exang"]})', layout=widgets.Layout(width='320px')),
    'oldpeak': widgets.BoundedFloatText(value=float(sample_patient['oldpeak']), min=float(raw_heart_df['oldpeak'].min()), max=float(raw_heart_df['oldpeak'].max()), step=0.1, description=f'Oldpeak ({feature_hints["oldpeak"]})', layout=widgets.Layout(width='320px')),
    'slope': widgets.Dropdown(options=make_dropdown_options(raw_heart_df['slope']), value=sample_patient['slope'], description=f'Slope ({feature_hints["slope"]})', layout=widgets.Layout(width='320px')),
    'ca': widgets.BoundedFloatText(value=float(sample_patient['ca']), min=float(raw_heart_df['ca'].min()), max=float(raw_heart_df['ca'].max()), step=1.0, description=f'CA ({feature_hints["ca"]})', layout=widgets.Layout(width='320px')),
    'thal': widgets.Dropdown(options=make_dropdown_options(raw_heart_df['thal']), value=sample_patient['thal'], description=f'Thal ({feature_hints["thal"]})', layout=widgets.Layout(width='320px')),
}

form_rows = [
    widgets.HBox([widget_specs['age'], widget_specs['sex']]),
    widgets.HBox([widget_specs['cp'], widget_specs['trestbps']]),
    widgets.HBox([widget_specs['chol'], widget_specs['fbs']]),
    widgets.HBox([widget_specs['restecg'], widget_specs['thalach']]),
    widgets.HBox([widget_specs['exang'], widget_specs['oldpeak']]),
    widgets.HBox([widget_specs['slope'], widget_specs['ca']]),
    widgets.HBox([widget_specs['thal']]),
]

predict_button = widgets.Button(
    description='Predict',
    button_style='danger',
    icon='heartbeat',
    layout=widgets.Layout(width='180px')
)

results_output = widgets.Output()


def preprocess_dashboard_input(values):
    input_frame = pd.DataFrame([values])[feature_order]
    encoded_frame = pd.get_dummies(input_frame, columns=categorical_cols, drop_first=True)
    encoded_frame = encoded_frame.reindex(columns=X_train.columns, fill_value=0)
    encoded_frame[continuous_cols] = scaler.transform(encoded_frame[continuous_cols])
    return encoded_frame


def render_prediction(results_container, values):
    processed_row = preprocess_dashboard_input(values)
    disease_probability = rf_best_model.predict_proba(processed_row)[0, 1]
    predicted_class = int(disease_probability >= 0.5)
    confidence = disease_probability if predicted_class == 1 else 1 - disease_probability

    top3_features = rf_importance.sort_values(ascending=False).head(3)
    top3_df = top3_features.reset_index()
    top3_df.columns = ['feature', 'importance']

    risk_label = 'High risk' if predicted_class == 1 else 'Low risk'
    risk_color = '#b91c1c' if predicted_class == 1 else '#166534'
    action_text = (
        'Recommend prompt cardiology review and follow-up testing.'
        if predicted_class == 1
        else 'No urgent disease signal, but continue routine monitoring and follow standard screening.'
    )

    with results_container:
        clear_output(wait=True)
        display(HTML(
            f"<div style='padding:12px;border-radius:10px;background:#f8fafc;border:1px solid #dbe4ee;'>"
            f"<div style='font-size:18px;font-weight:700;color:{risk_color};'>Predicted class: {risk_label}</div>"
            f"<div style='font-size:15px;margin-top:6px;'>Confidence: {confidence * 100:.1f}%</div>"
            f"<div style='font-size:14px;margin-top:6px;'>Model probability of disease: {disease_probability:.3f}</div>"
            f"</div>"
        ))

        plt.figure(figsize=(6.5, 3.8))
        sns.barplot(data=top3_df, x='importance', y='feature', color='steelblue')
        plt.xlabel('Feature importance')
        plt.ylabel('Feature')
        plt.title('Top 3 Driving Features')
        plt.tight_layout()
        plt.show()

        display(HTML(
            f"<div style='margin-top:8px;padding:10px;border-left:4px solid {risk_color};background:#ffffff;'>"
            f"<strong>Nurse note:</strong> {action_text}"
            f"</div>"
        ))

        display(HTML(
            "<div style='margin-top:8px;color:#334155;'>"
            f"Top contributors: {', '.join(top3_features.index.tolist())}."
            "</div>"
        ))



def on_predict_clicked(_):
    values = {name: widget_specs[name].value for name in feature_order}
    render_prediction(results_output, values)


predict_button.on_click(on_predict_clicked)

feature_help = widgets.HTML(
    "<div style='padding:8px 0 12px 0;'>"
    "<strong>Notebook dashboard:</strong> edit the inputs below, then press Predict. "
    "The form is pre-populated from a real patient row from the cleaned heart-disease dataset."
    "</div>"
)

form_ui = widgets.VBox([
    feature_help,
    *form_rows,
    predict_button,
    results_output,
])

display(form_ui)

# Render the default prediction immediately so the dashboard has an initial result.
on_predict_clicked(None)
