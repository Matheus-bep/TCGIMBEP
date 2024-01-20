from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


num_iterations = 1000

data_original = pd.read_csv('Dados Originais.csv')
data_tratado = pd.read_csv('Dados Tratados.csv')


def select_variables(data, outcome):
    excluded_vars = []

    if outcome == 'result':
        if 'league' in data.columns:
            data = data.drop('league', axis=1, errors='ignore')
            excluded_vars.append('league')
    elif outcome == 'league':
        if 'result' in data.columns:
            data = data.drop('result', axis=1, errors='ignore')
            excluded_vars.append('result')

    corr_matrix = data.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    data = data.drop(to_drop, axis=1)
    excluded_vars.extend(to_drop)

    selected_var = []

    for column in data.drop(outcome, axis=1):
        X = data[[column]]
        y = data[outcome]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

        if model.pvalues[1] < model.pvalues[0]:
            selected_var.append(column)
        else:
            excluded_vars.append(column)

    print("Variaveis Excluidas:", excluded_vars)
    return selected_var

def run_random_forest(data, outcome, selected_var):
    accuracy_list = []
    precision_list = []
    f1_list = []
    recall_list = []
    roc_auc_list = []

    for _ in range(num_iterations):
        data_sampled = resample(data, replace=True, n_samples=len(data))

        # Verificar se as colunas selecionadas existem nos dados
        selected_var_existing = [col for col in selected_var if col in data_sampled.columns]

        if not selected_var_existing:
            raise ValueError("Nenhuma variavel selecionada disponivel nos dados.")

        X = data_sampled[selected_var_existing]
        y = data_sampled[outcome]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Random Forest
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        roc_auc_list.append(roc_auc)

    accuracy_ci = np.percentile(accuracy_list, [2.5, 97.5])
    precision_ci = np.percentile(precision_list, [2.5, 97.5])
    recall_ci = np.percentile(recall_list, [2.5, 97.5])
    f1_ci = np.percentile(f1_list, [2.5, 97.5])
    roc_auc_ci = np.percentile(roc_auc_list, [2.5, 97.5])

    return {
        'accuracy': np.mean(accuracy_list),
        'precision': np.mean(precision_list),
        'recall': np.mean(recall_list),
        'f1': np.mean(f1_list),
        'roc_auc': np.mean(roc_auc_list),
        'accuracy_ci': accuracy_ci,
        'precision_ci': precision_ci,
        'recall_ci': recall_ci,
        'f1_ci': f1_ci,
        'roc_auc_ci': roc_auc_ci
    }

def run_logistic_regression(data, outcome, selected_var):
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []

    for _ in range(num_iterations):
        data_sampled = resample(data, replace=True, n_samples=len(data))

        # Verificar se as colunas selecionadas existem nos dados
        selected_var_existing = [col for col in selected_var if col in data_sampled.columns]

        if not selected_var_existing:
            raise ValueError("Nenhuma variavel selecionada disponivel nos dados.")

        X = data_sampled[selected_var_existing]
        y = data_sampled[outcome]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        roc_auc_list.append(roc_auc)

    accuracy_ci = np.percentile(accuracy_list, [2.5, 97.5])
    precision_ci = np.percentile(precision_list, [2.5, 97.5])
    recall_ci = np.percentile(recall_list, [2.5, 97.5])
    f1_ci = np.percentile(f1_list, [2.5, 97.5])
    roc_auc_ci = np.percentile(roc_auc_list, [2.5, 97.5])

    return {
        'accuracy': np.mean(accuracy_list),
        'precision': np.mean(precision_list),
        'recall': np.mean(recall_list),
        'f1': np.mean(f1_list),
        'roc_auc': np.mean(roc_auc_list),
        'accuracy_ci': accuracy_ci,
        'precision_ci': precision_ci,
        'recall_ci': recall_ci,
        'f1_ci': f1_ci,
        'roc_auc_ci': roc_auc_ci
    }

def plot_roc_curve(y_true, y_score, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(filename)   
    

# Substituir valores nulos pela mediana para todo o dataset
data_original = data_original.fillna(data_original.median())
data_tratado = data_tratado.fillna(data_tratado.median())

# Executar para 'result' no dataset 'original'
selected_var_result_original = select_variables(data_original, 'result')
metrics_result_original_rf = run_random_forest(data_original, 'result', selected_var_result_original)
y_score_result_original_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3).fit(
    data_original[selected_var_result_original], data_original['result']).predict_proba(data_original[selected_var_result_original])[:, 1]

metrics_result_original_lr = run_logistic_regression(data_original, 'result', selected_var_result_original)
y_score_result_original_lr = LogisticRegression(random_state=42).fit(
    data_original[selected_var_result_original], data_original['result']).predict_proba(data_original[selected_var_result_original])[:, 1]

# Executar para 'league' no dataset 'original'
selected_var_league_original = select_variables(data_original, 'league')
metrics_league_original_rf = run_random_forest(data_original, 'league', selected_var_league_original)
y_score_league_original_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3).fit(
    data_original[selected_var_league_original], data_original['league']).predict_proba(data_original[selected_var_league_original])[:, 1]

metrics_league_original_lr = run_logistic_regression(data_original, 'league', selected_var_league_original)
y_score_league_original_lr = LogisticRegression(random_state=42).fit(
    data_original[selected_var_league_original], data_original['league']).predict_proba(data_original[selected_var_league_original])[:, 1]

# Executar para 'result' no dataset 'tratado'
selected_var_result_tratado = select_variables(data_tratado, 'result')
metrics_result_tratado_rf = run_random_forest(data_tratado, 'result', selected_var_result_tratado)
y_score_result_tratado_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3).fit(
    data_tratado[selected_var_result_tratado], data_tratado['result']).predict_proba(data_tratado[selected_var_result_tratado])[:, 1]

metrics_result_tratado_lr = run_logistic_regression(data_tratado, 'result', selected_var_result_tratado)
y_score_result_tratado_lr = LogisticRegression(random_state=42).fit(
    data_tratado[selected_var_result_tratado], data_tratado['result']).predict_proba(data_tratado[selected_var_result_tratado])[:, 1]

# Executar para 'league' no dataset 'tratado'
selected_var_league_tratado = select_variables(data_tratado, 'league')
metrics_league_tratado_rf = run_random_forest(data_tratado, 'league', selected_var_league_tratado)
y_score_league_tratado_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3).fit(
    data_tratado[selected_var_league_tratado], data_tratado['league']).predict_proba(data_tratado[selected_var_league_tratado])[:, 1]

metrics_league_tratado_lr = run_logistic_regression(data_tratado, 'league', selected_var_league_tratado)
y_score_league_tratado_lr = LogisticRegression(random_state=42).fit(
    data_tratado[selected_var_league_tratado], data_tratado['league']).predict_proba(data_tratado[selected_var_league_tratado])[:, 1]


print("Metricas para 'result' no dataset 'original' (Random Forest):", metrics_result_original_rf)
print("Metricas para 'result' no dataset 'tratado' (Random Forest):", metrics_result_tratado_rf)
print("Metricas para 'league' no dataset 'original' (Random Forest):", metrics_league_original_rf)
print("Metricas para 'league' no dataset 'tratado' (Random Forest):", metrics_league_tratado_rf)

print("Metricas para 'result' no dataset 'original' (Regressao Logistica):", metrics_result_original_lr)
print("Metricas para 'result' no dataset 'tratado' (Regressao Logistica):", metrics_result_tratado_lr)
print("Metricas para 'league' no dataset 'original' (Regressao Logistica):", metrics_league_original_lr)
print("Metricas para 'league' no dataset 'tratado' (Regressao Logistica):", metrics_league_tratado_lr)

plot_roc_curve(data_original['result'], y_score_result_original_rf, 'ROC-AUC Curve - RF Result Original', 'ROC-AUC - RF Result Original')
plot_roc_curve(data_original['league'], y_score_league_original_rf, 'ROC-AUC Curve - RF League Original', 'ROC-AUC - RF League Original')
plot_roc_curve(data_tratado['result'], y_score_result_tratado_rf, 'ROC-AUC Curve - RF Result Tratado', 'ROC-AUC - RF Result Tratado')
plot_roc_curve(data_tratado['league'], y_score_league_tratado_rf, 'ROC-AUC Curve - RF League Tratado', 'ROC-AUC - RF League Tratado')

plot_roc_curve(data_original['result'], y_score_result_original_lr, 'ROC-AUC Curve - RL Result Original', 'ROC-AUC - RL Result Original')
plot_roc_curve(data_original['league'], y_score_league_original_lr, 'ROC-AUC Curve - RL League Original', 'ROC-AUC - RL League Original')
plot_roc_curve(data_tratado['result'], y_score_result_tratado_lr, 'ROC-AUC Curve - RL Result Tratado', 'ROC-AUC - RL Result Tratado')
plot_roc_curve(data_tratado['league'], y_score_league_tratado_lr, 'ROC-AUC Curve - RL League Tratado', 'ROC-AUC - RL League Tratado')
