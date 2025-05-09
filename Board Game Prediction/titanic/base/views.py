# base/views.py
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import BytesIO
import base64

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.conf import settings

# --- Load your ML models ---
with open('base/GridSearchCV', 'rb') as f: # This is your best_rf_model from GridSearchCV
    rf_model_grid = pickle.load(f)

with open('base/RandomizedSearchCV', 'rb') as f: # This is your best_rf_random_model
    rf_model_random = pickle.load(f)

# --- Helper function to generate and save charts ---
def generate_chart_as_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

# --- Homepage / Landing Page / Dashboard View ---
def landing_page(request):
    context = {}
    if request.user.is_authenticated:
        context['is_dashboard'] = True
        try:
            csv_path = os.path.join(os.path.dirname(__file__), 'Project.csv')
            if not os.path.exists(csv_path):
                 csv_path = os.path.join(settings.BASE_DIR, 'Project.csv')
            df = pd.read_csv(csv_path)

            # 1. Dataset Statistics (Same as before)
            context['num_records'] = df.shape[0]
            context['num_features'] = df.shape[1]
            numeric_cols_for_summary = ['Year Published', 'Users Rated', 'Rating Average', 'BGG Rank', 'Complexity Average', 'Owned Users']
            existing_summary_cols = [col for col in numeric_cols_for_summary if col in df.columns]
            if existing_summary_cols:
                desc_stats = df[existing_summary_cols].describe().to_html(classes='table table-sm table-striped table-bordered', border=0)
                context['feature_summary_html'] = desc_stats
            else:
                context['feature_summary_html'] = "<p>Feature summary could not be generated.</p>"

            fig_target, ax_target = plt.subplots(figsize=(8, 5))
            sns.histplot(df['Rating Average'], kde=True, ax=ax_target, bins=15)
            ax_target.set_title('Distribution of Rating Average')
            ax_target.set_xlabel('Rating Average')
            ax_target.set_ylabel('Frequency')
            context['target_distribution_chart'] = generate_chart_as_base64(fig_target)

            # 2. Model Performance
            # Performance values from your IPYNB (Worked.ipynb)
            context['model_performance'] = [ # Changed to a list
                {
                    'model_name': 'Random Forest (RandomizedSearchCV Optimized)', # Cell 42
                    'mae': 0.0588,
                    'rmse': 0.0814,
                    'r_squared': 0.9312,
                },
                {
                    'model_name': 'Random Forest (GridSearchCV Optimized)', # Cell 41
                    'mae': 0.0644,
                    'rmse': 0.0863,
                    'r_squared': 0.9226,
                }
            ]

            # 3. Charts
            # Correlation Heatmap (Same as before)
            df_numeric_only = df.select_dtypes(include=np.number)
            if not df_numeric_only.empty:
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_numeric_only.corr(), annot=False, cmap="coolwarm", ax=ax_corr, fmt=".2f")
                ax_corr.set_title('Correlation Heatmap (Numeric Features)')
                context['correlation_heatmap_chart'] = generate_chart_as_base64(fig_corr)
            else:
                context['correlation_heatmap_chart'] = None
            
            # --- Feature Importances ---
            # Re-define features X as used for training to get column names
            processed_df = df.drop(columns=['Rating Average', 'ID', 'Name'], errors='ignore')
            if 'Mechanics' in processed_df.columns:
                processed_df['Mechanics Count'] = processed_df['Mechanics'].apply(lambda x: len(str(x).split(',')))
                processed_df = processed_df.drop(columns=["Mechanics"])
            if 'Domains' in processed_df.columns:
                processed_df['Domains Count'] = processed_df['Domains'].apply(lambda x: len(str(x).split(',')))
                processed_df = processed_df.drop(columns=["Domains"])
            X_for_importance = processed_df.select_dtypes(include=np.number)

            charts_feature_importance = []

            # Feature Importance for RandomizedSearchCV model
            if hasattr(rf_model_random, 'feature_importances_') and not X_for_importance.empty:
                importances_random = rf_model_random.feature_importances_
                if len(importances_random) == len(X_for_importance.columns):
                    feature_names = X_for_importance.columns
                    fi_df_random = pd.DataFrame({'Feature': feature_names, 'Importance': importances_random})
                    fi_df_random = fi_df_random.sort_values(by='Importance', ascending=False).head(10)

                    fig_random, ax_random = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=fi_df_random, ax=ax_random, palette="viridis_r", hue='Feature', legend=False)
                    ax_random.set_title('Top Feature Importances (RandomizedSearchCV)')
                    charts_feature_importance.append({
                        'title': 'RandomizedSearchCV Model',
                        'chart_data': generate_chart_as_base64(fig_random)
                    })
                else:
                    charts_feature_importance.append({'title': 'RandomizedSearchCV Model', 'chart_data': '<p>Error: Feature count mismatch.</p>'})
            else:
                charts_feature_importance.append({'title': 'RandomizedSearchCV Model', 'chart_data': '<p>Feature importance data not available.</p>'})

            # Feature Importance for GridSearchCV model
            if hasattr(rf_model_grid, 'feature_importances_') and not X_for_importance.empty:
                importances_grid = rf_model_grid.feature_importances_
                if len(importances_grid) == len(X_for_importance.columns):
                    feature_names = X_for_importance.columns # Same feature names
                    fi_df_grid = pd.DataFrame({'Feature': feature_names, 'Importance': importances_grid})
                    fi_df_grid = fi_df_grid.sort_values(by='Importance', ascending=False).head(10)

                    fig_grid, ax_grid = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=fi_df_grid, ax=ax_grid, palette="mako_r", hue='Feature', legend=False)
                    ax_grid.set_title('Top Feature Importances (GridSearchCV)')
                    charts_feature_importance.append({
                        'title': 'GridSearchCV Model',
                        'chart_data': generate_chart_as_base64(fig_grid)
                    })
                else:
                    charts_feature_importance.append({'title': 'GridSearchCV Model', 'chart_data': '<p>Error: Feature count mismatch.</p>'})
            else:
                charts_feature_importance.append({'title': 'GridSearchCV Model', 'chart_data': '<p>Feature importance data not available.</p>'})
            
            context['feature_importance_charts'] = charts_feature_importance


        except FileNotFoundError:
            messages.error(request, "Dataset file 'Project.csv' not found.")
            context['dashboard_error'] = "Dataset file not found. Cannot display dashboard statistics."
        except Exception as e:
            messages.error(request, f"An error occurred while generating dashboard data: {str(e)}")
            context['dashboard_error'] = f"An error occurred: {str(e)}"
            print(f"Dashboard generation error: {e}")

    return render(request, 'landing_page.html', context)

# --- Prediction Tool Views (Ensure redirects point to 'prediction_tool' not 'index') ---
@login_required
def prediction_tool_view(request):
    error_message = request.session.pop('error_message', None)
    return render(request, 'index.html', {'error_message': error_message})

@login_required
def predict(request):
    if request.method == 'POST':
        try:
            year_published = int(request.POST['year_published'])
            users_rated = int(request.POST['users_rated'])
            bgg_rank = int(request.POST['bgg_rank'])
            owned_users = int(request.POST['owned_users'])
            mechanics_count = int(request.POST['mechanics_count'])
            model_choice = request.POST.get('model_choice', 'grid')

            input_data = np.array([[
                year_published, 2, 4, 60, 10, users_rated,
                bgg_rank, 2.5, owned_users, mechanics_count, 2
            ]])

            if model_choice == 'random':
                prediction = rf_model_random.predict(input_data)[0]
                model_used_str = 'RandomizedSearchCV'
            else:
                prediction = rf_model_grid.predict(input_data)[0]
                model_used_str = 'GridSearchCV'

            return render(request, 'result.html', {
                'prediction': round(prediction, 2),
                'model_used': model_used_str
            })
        except ValueError:
            request.session['error_message'] = "Invalid input. Please ensure all fields are numbers."
            return redirect('prediction_tool') 
        except Exception as e:
            request.session['error_message'] = f"An error occurred: {str(e)}"
            return redirect('prediction_tool') 

    return redirect('prediction_tool')

# --- Registration View ---
def register(request):
    if request.user.is_authenticated:
        return redirect('landing_page')

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Registration successful! Please log in with your new account.')
            return redirect('login')
        else:
            for field in form:
                for error in field.errors:
                    messages.error(request, f"{field.label}: {error}")
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})