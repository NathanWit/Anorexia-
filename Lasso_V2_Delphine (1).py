#!/usr/bin/env python
# coding: utf-8

# # Lasso

# ## Import librairies

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
import category_encoders as ce
get_ipython().run_line_magic('matplotlib', 'inline')

# import Random Forest classifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import os


# In[2]:


import warnings

warnings.filterwarnings('ignore')


# In[3]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 238)


# In[4]:


sns.set_context("notebook")


# ## Data importations

# In[5]:


# Delphine 
df = pd.read_csv(r"/Users/delphinepaquiry/Documents/Mines /EtudesTechniques/Data.csv")

# Nathan 
#df = pd.read_csv(r"C:\Users\Nathan\Desktop\Etude technique\Data.csv")


# In[13]:


df['Duréemoisdepuisdernièresmenstruations']


# ### Exploratory data analysis

# In[15]:


# view dimensions of dataset
df.shape


# In[63]:


# preview the dataset
df.tail(100)


# ### Check filling the columns

# In[17]:


df.isnull().sum()


# ### Others Binary Columns

# In[19]:


binary_columns= [
    "AnorexiedetypeboulemieActuel",
    "utilisationActuellaxatif",
    "utilisationActuelDiurétique",
    "AnorexiedetyperestrictifActuel",
    "AnorexiedetypeboulemiePasse",
    "utilisationPasselaxatif",
    "AnorexiedetyperestrictifPasse",
    "amenorheprimaire",
    "traitementcontraceptif",
    "Cyclesréguliers",
    "Signehyperandrogenie", 
    "Hyperactivite",
    "ActivitePhysiqueActuelPratriquee1 h/s",
    "ActivitePhysiquePassePratriquee1"    
]


# In[20]:


df = df.drop(columns=binary_columns)


# ### Create Column with date

# In[21]:


colonnes_date =[
    'Dateexam',
    'DDN',
    'datePoidsleplusbas',
    'datePoidsleplushaut',
    'DatedebutTCA'
]


# #### Convertir les colonnes Dates

# In[22]:


df['Dateexam'] = pd.to_datetime(df['Dateexam'])
df['datePoidsleplusbas'] = pd.to_datetime(df['datePoidsleplusbas'])
df['datePoidsleplushaut'] = pd.to_datetime(df['datePoidsleplushaut'])
df['DatedebutTCA'] = pd.to_datetime(df['DatedebutTCA'])


# #### Calculer les différences

# In[23]:


df['MoisPoidsBasExam'] = (df['Dateexam'].dt.year - df['datePoidsleplusbas'].dt.year) * 12 + (df['Dateexam'].dt.month - df['datePoidsleplusbas'].dt.month)

df['MoisPoidsHautExam'] = (df['Dateexam'].dt.year - df['datePoidsleplushaut'].dt.year) * 12 + (df['Dateexam'].dt.month - df['datePoidsleplushaut'].dt.month)

df['MoisDebutTCAExam'] = (df['Dateexam'].dt.year - df['DatedebutTCA'].dt.year) * 12 + (df['Dateexam'].dt.month - df['DatedebutTCA'].dt.month)


# #### Afficher les nouvelles colonnes

# In[24]:


df = df.drop(columns=colonnes_date)


# In[25]:


df


# ### Declare feature vector and target variable

# In[26]:


X = df.drop(columns=['L1L4ZscoreSD'])
y = df['L1L4ZscoreSD']


# In[27]:


np.mean(y)


# ### Normaliation 

# In[28]:


# Initialiser le MinMaxScaler
scaler = StandardScaler()

# Normaliser les données
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# ### Split data into separate training and test set 

# In[29]:


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[30]:


# check the shape of X_train and X_test
X_train.shape, X_test.shape


# In[31]:


colonnes_non_numeriques = df.select_dtypes(exclude=['int', 'float']).columns

# Afficher les noms des colonnes non numériques
colonnes_non_numeriques


# ### Choose of alpha

# #### Définition des valeurs d'alpha à tester

# In[32]:


alphas = np.arange(0, 1, 0.01)


# #### Liste pour stocker le MSE et le nombre de caractéristiques nulles pour chaque alpha

# In[33]:


mse_values_train = []
mse_values_test = []
zero_features_count = []
non_zero_features_count = []


# #### Creation de la boucle de test

# In[34]:


for alpha in alphas:
    #print(alpha)
    # Création du modèle Lasso
    model = Lasso(alpha=alpha, random_state=42)
    
    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Calcul de l'erreur quadratique moyenne (MSE)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mse_values_train.append(mse_train)
    mse_values_test.append(mse_test)

    # Comptage des caractéristiques nulles
    zero_features = np.sum(model.coef_ == 0)
    zero_features_count.append(zero_features)
    
    non_zero_features_count.append(95 - zero_features)
    
    #print(model.coef_)


# In[35]:


help(Lasso)


# In[36]:


print(non_zero_features_count)


# #### Tracé des courbes Alpha vs MSE

# In[37]:


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Alpha')
ax1.set_ylabel('Mean Squared Error Train (MSE)', color=color)
ax1.plot(alphas, mse_values_train, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Nombre de features non nulles', color=color)
ax2.plot(alphas, non_zero_features_count, color=color)
ax2.tick_params(axis='y', labelcolor=color)


fig.tight_layout()
plt.title('Lasso: Alpha vs MSE train & Nombre de features non nulles')

plt.show()


# In[ ]:





# In[38]:


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Alpha')
ax1.set_ylabel('Mean Squared Error Test (MSE)', color=color)
ax1.plot(alphas, mse_values_test, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Nombre de features non nulles', color=color)
ax2.plot(alphas, non_zero_features_count, color=color)
ax2.tick_params(axis='y', labelcolor=color)


fig.tight_layout()
plt.title('Lasso: Alpha vs MSE test & Nombre de features nulles')
plt.show()


# In[39]:


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Alpha')
ax1.set_ylabel('Mean Squared Error Train (MSE)', color=color)
ax1.plot(alphas, mse_values_train, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Mean Squared Error Test (MSE)', color=color)
ax2.plot(alphas, mse_values_test, color=color)
ax2.tick_params(axis='y', labelcolor=color)


fig.tight_layout()
plt.title('Lasso: Alpha vs MSE train & MSE test')

plt.show()


# #### Autres types de courbes

# In[42]:


plt.figure(figsize=(10, 6))
plt.plot(zero_features_count, alphas, marker='o', linestyle='-')
plt.title('Nombre de caractéristiques non nulles vs Alpha (Lasso)')
plt.ylabel('Alpha')
plt.xlabel('Nombre de caractéristiques non nulles')
plt.grid(True)
plt.show()


# In[43]:


data = {'zero_features_count': zero_features_count, 'alphas': alphas, 'mse':mse_values_train}
zero_features_count_df = pd.DataFrame(data)
zero_features_count_df


# In[44]:


alpha = 0.08


# ### Lasso Model

# In[45]:


prediction = []


# In[46]:


lasso_model = Lasso(alpha=alpha, random_state=42) 
lasso_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
predictions = lasso_model.predict(X_test)


# In[47]:


import numpy as np

# Calculer l'erreur
errors = np.abs(y_test - predictions)



data_resultats = {'errors': errors, 'value_test':y_test, 'predictions':predictions}
useful = pd.DataFrame(data_resultats)


# Supposons que errors soit une colonne dans votre DataFrame df
# Vous pouvez utiliser la méthode sort_values pour trier le DataFrame selon la colonne errors
df_sorted = useful.sort_values(by='errors', ascending=False)

# Afficher le DataFrame trié
print(df_sorted)


# In[48]:


# Tracer l'histogramme des pourcentages d'erreur
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=20, color='green', edgecolor='black')
plt.xlabel('Erreur absolue')
plt.ylabel('Nombre de prédictions')
plt.grid(True)
plt.show()


# In[49]:


import matplotlib.pyplot as plt

# Tracer les valeurs prédites par le modèle par rapport aux valeurs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_train, lasso_model.predict(X_train), color='blue', label="Valeurs prédites pour l'ensemble train")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
plt.title('Comparaison des valeurs prédites et des valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.legend()

plt.show()


# In[50]:


import matplotlib.pyplot as plt

# Tracer les valeurs prédites par le modèle par rapport aux valeurs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', label='Valeurs prédites')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
plt.title('Comparaison des valeurs prédites et des valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.legend()
plt.show()


# ### Analyse du modèle

# In[51]:


print(lasso_model.intercept_)


# ### Calcul de l'erreur moyenne quadratique (RMSE)

# In[52]:


mse = mean_squared_error(y_test, predictions, squared=False)
print("MSE:", mse)


# ### Histogramme des coefficients

# In[53]:


plt.figure(figsize=(10, 10))  
w = lasso_model.coef_
sns.histplot(w, bins=len(w), kde=False)  
plt.title('Distribution des coefficients du modèle Lasso')
plt.xlabel('Valeur des coefficients')
plt.ylabel('Fréquence')
plt.show()


# ### Find important features with Random Forest model 

# In[54]:


# Récupération des coefficients attribués à chaque caractéristique
lasso_coefs = lasso_model.coef_

# Création d'un dataframe pour visualiser les coefficients des caractéristiques
lasso_coefs_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso_coefs
})

# Trier le dataframe par valeur absolue des coefficients
lasso_coefs_df['AbsoluteCoefficient'] = abs(lasso_coefs_df['Coefficient'])
lasso_coefs_df = lasso_coefs_df.sort_values(by='AbsoluteCoefficient', ascending=False)

# Affichage des caractéristiques importantes
print(lasso_coefs_df.head(15))


# In[55]:


top_15_features = lasso_coefs_df['Feature'].head(15).tolist()


# ### Correlation Matrix

# In[56]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Calculer la matrice de corrélation
correlation_matrix = df.corr()

# Définir le seuil de corrélation
correlation_threshold = 0.90

# Filtrer la matrice de corrélation pour ne montrer que les valeurs au-dessus du seuil
high_correlation_pairs = np.where(np.abs(correlation_matrix) > correlation_threshold)

# Récupérer les indices des colonnes corrélées
correlated_columns_indices = set()
for col1, col2 in zip(*high_correlation_pairs):
    if col1 != col2 and col1 not in correlated_columns_indices and col2 not in correlated_columns_indices:
        correlated_columns_indices.add(col1)
        correlated_columns_indices.add(col2)

# Extraire les noms des colonnes corrélées
correlated_columns = df.columns[list(correlated_columns_indices)]

# Créer une nouvelle matrice de corrélation avec seulement les colonnes corrélées
subset_correlation_matrix = df[correlated_columns].corr()

# Créer une figure et un axe
plt.figure(figsize=(12, 10))

# Tracer la matrice de corrélation avec seaborn
sns.heatmap(subset_correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")

# Afficher le graphique
plt.title('Matrice de corrélation entre les colonnes fortement corrélées')
plt.show()


# ### Tests sur d'autres modeles avec les 15 features les plus importantes

# In[57]:


df.shape


# In[58]:


df_reduit = df[top_15_features]
df_reduit.info()


# In[59]:


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

X_train, X_test, y_train, y_test = train_test_split(df_reduit, y, test_size=0.3, random_state=42)

# Initialiser les modèles de régression
linear_reg_model = LinearRegression()
ridge_model = Ridge()
gradient_boosting_model = GradientBoostingRegressor()
xgboost_model = XGBRegressor()
knn_model = KNeighborsRegressor()
svr_model = SVR()

# Liste des modèles
models = [linear_reg_model, ridge_model,
          gradient_boosting_model,
          xgboost_model, knn_model, svr_model]

# Boucle d'entraînement et d'évaluation des modèles
for model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Calculer l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(y_test, predictions)
    
    # Afficher le MSE pour chaque modèle
    print(f"{model.__class__.__name__} - MSE: {mse}")

    # Vous pouvez également visualiser les prédictions par rapport aux vraies valeurs
    plt.scatter(y_test, predictions)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()


# In[60]:


gradient_boosting_model.fit(X_train, y_train)
predictions_gb = gradient_boosting_model.predict(X_test)

# Calculer l'erreur quadratique moyenne (MSE)
mse = mean_squared_error(y_test, predictions_gb)

# Afficher le MSE pour chaque modèle
print(f"{gradient_boosting_model.__class__.__name__} - MSE: {mse}")

# Vous pouvez également visualiser les prédictions par rapport aux vraies valeurs
plt.scatter(y_test, predictions_gb)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()


# In[61]:


# Calculer l'erreur
errors_gb = np.abs(y_test - predictions_gb)


data = {'errors': errors_gb, 'value_test':y_test, 'predictions':predictions_gb}
useful = pd.DataFrame(data)


# Supposons que errors soit une colonne dans votre DataFrame df
# Vous pouvez utiliser la méthode sort_values pour trier le DataFrame selon la colonne errors
df_sorted = useful.sort_values(by='errors', ascending=False)


# Tracer l'histogramme des pourcentages d'erreur
plt.figure(figsize=(10, 6))
plt.hist(errors_gb, bins=20, color='green', edgecolor='black')
plt.xlabel('Erreur absolue')
plt.ylabel('Nombre de prédictions')
plt.title("Erreur de prédictions pour le GradientBoosting Model")
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




