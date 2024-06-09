#!/usr/bin/env python
# coding: utf-8

# # Hola &#x1F600;
# 
# Soy **Hesus Garcia**, revisor de código de Triple Ten, y voy a examinar el proyecto que has desarrollado recientemente. Si encuentro algún error, te lo señalaré para que lo corrijas, ya que mi objetivo es ayudarte a prepararte para un ambiente de trabajo real, donde el líder de tu equipo actuaría de la misma manera. Si no puedes solucionar el problema, te proporcionaré más información en la próxima oportunidad. Cuando encuentres un comentario,  **por favor, no los muevas, no los modifiques ni los borres**. 
# **Una gran disculpa por el retraso en la revisión de tu proyecto. Hemos tenido una carga de proyectos que nos sobrepasó**
# Revisaré cuidadosamente todas las implementaciones que has realizado para cumplir con los requisitos y te proporcionaré mis comentarios de la siguiente manera:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si todo está perfecto.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# Puedes responderme de esta forma:
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
# </div>
# 
# </br>
# 
# **¡Empecemos!**  &#x1F680;
# 

# # INTRODUCCION

# El siguiente proyecto consiste en asesorar a la compañía de telecomunicaciones Interconnect sobre la creación de un modelo predictivo de la tasa de cancelación de clientes, de tal forma de que la compañía pueda anticiparse para ofrecer códigos promocionlaes y opciones de planes especiales. 
# 
# Para ello se utilizaran bases de datos facilitadas por el equipo de marketing de Interconnect que recopila datos desde el 2016 en adelante.

# ## Inicialización

# In[1]:


import pandas as pd
import re
from scipy import stats as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.metrics import roc_curve, roc_auc_score, auc


# ## Carga de datos

# In[2]


contract = pd.read_csv('/Users/rmmniv/Library/Mobile Documents/com~apple~CloudDocs/Data Science/SPRINT 17 - Proyecto Final/final_provider/contract.csv')
personal = pd.read_csv('/Users/rmmniv/Library/Mobile Documents/com~apple~CloudDocs/Data Science/SPRINT 17 - Proyecto Final/final_provider/personal.csv')
internet = pd.read_csv('/Users/rmmniv/Library/Mobile Documents/com~apple~CloudDocs/Data Science/SPRINT 17 - Proyecto Final/final_provider/internet.csv')
phone = pd.read_csv('/Users/rmmniv/Library/Mobile Documents/com~apple~CloudDocs/Data Science/SPRINT 17 - Proyecto Final/final_provider/phone.csv')


# <div class="alert alert-block alert-danger">
#     <b>Comentarios del Revisor</b> <a class="tocSkip"></a><br>
# Me parece que estos no son los datasets para el proyecto final, por favor revisa mi comentario del final.  </div>

# 
# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante</b> <a class=“tocSkip”></a>
# 
#     Corregido el llaado a los datasets
# </div>
# 

# ## Análisis exploratorio de datos (EDA)

# In[3]:


contract.info() #Verificamos los campos a nivel general de 'contract'


# Glosario de dataset 'contract':
# 
# * customerID - ID del cliente (clave para unir datasets).
# * BeginDate - Fecha de inicio del contrato (puede ayudar a calcular la duración del contrato).
# * EndDate - Fecha de fin del contrato (puede ayudar a determinar si un contrato ha finalizado recientemente).
# * Type - Tipo de contrato (mensual, 1 año, 2 años), importante para entender la fidelidad del cliente.
# * PaperlessBilling - Facturación electrónica (puede influir en la satisfacción del cliente).
# * PaymentMethod - Método de pago (algunos métodos pueden estar asociados con una mayor probabilidad de churn).
# * MonthlyCharges - Cobro mensual (importante para entender la carga financiera sobre el cliente).
# * TotalCharges - Cobro total (puede reflejar el valor del cliente a lo largo del tiempo).

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# La carga de datos se realiza correctamente, utilizando pandas para leer los archivos CSV. Además, se incluye una descripción de los campos en el dataset 'contract', lo que facilita la comprensión de las variables disponibles para el análisis.
# </div>

# In[4]:


contract.tail(4) #Vemos una muestra


# In[5]:


#Verificamos duplicidad
duplicados1 = contract.duplicated()
cantidad_duplicados1 = duplicados1.sum()
cantidad_duplicados1 # Finalmente, comprobamos el número de filas duplicadas de 'contract'


# A continuacion procedemos a revisar los posibles errores y/o anomalías de cada variable de 'contract'.

# In[6]:


contract['BeginDate'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en las fechas.


# In[7]:


contract['EndDate'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en las fechas.


# In[8]:


# Creamos la variable objetivo en una nueva columna, ya que es clave para el análisis.
contract['ContractStatus'] = contract['EndDate'].apply(lambda x: 1 if x == 'No' else 0)

# Verificamos el resultado
print(contract[['EndDate', 'ContractStatus']].head())


# In[9]:


contract['Type'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en los datos.


# In[10]:


contract['PaperlessBilling'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en los datos.


# In[11]:


contract['PaymentMethod'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en los datos.


# In[12]:


personal.info() #Verificamos los campos a nivel general de 'personal'


# Glosario para dataset 'personal':
# 
# * customerID - ID del cliente.
# * gender - Género (podría tener alguna correlación con la tasa de cancelación, aunque debe analizarse cuidadosamente para evitar sesgos).
# * SeniorCitizen - Indicador de si el cliente es ciudadano senior (los clientes mayores pueden tener diferentes patrones de comportamiento).
# * Partner - Si el cliente tiene pareja (puede influir en la estabilidad del cliente con el servicio).
# * Dependents - Si el cliente tiene dependientes (podría influir en la decisión de cancelar el servicio).

# In[13]:


personal.head(4)  # Vemos una muestra


# In[14]:


#Verificamos duplicidad
duplicados2 = personal.duplicated()
cantidad_duplicados2 = duplicados2.sum()
cantidad_duplicados2 # Finalmente, comprobamos el número de filas duplicadas de 'personal'


# A continuacion procedemos a revisar los posibles errores y/o anomalías de cada variable de 'personal'.

# In[15]:


personal['gender'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en los datos.


# In[16]:


personal['SeniorCitizen'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en los datos.


# In[17]:


personal['Partner'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en los datos.


# In[18]:


personal['Dependents'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en los datos.


# In[19]:


internet.info() #Verificamos los campos a nivel general de 'internet'


# Glosario del dataset 'internet':
#      
# * customerID: ID del cliente.
# * InternetService: Servicios de internet a través de línea telefónica (DSL, línea de abonado digital) o a través de un cable de fibra óptica.
# * OnlineSecurity: Si tiene o no servicio de bloqueador de sitios web maliciosos
# * OnlineBackup: Si tiene o no el servicio de almacenamiento y back up de datos.
# * DeviceProtection: Si tiene el servicio o no de antivirus.
# * TechSupport: Si tiene servicio de soporte técnico.
# * StreamingTV: Si tiene o no servicio de TV.
# * StreamingMovies: Si tiene o no servicio de directorio de peliculas.
# 

# In[20]:


internet.tail(4) #Vemos una muestra


# In[21]:


#Verificamos duplicidad
duplicados3 = internet.duplicated()
cantidad_duplicados3 = duplicados3.sum()
cantidad_duplicados3 # Finalmente, comprobamos el número de filas duplicadas de 'internet'


# In[22]:


internet.describe() # Vemos si hay anomalías en los valores numéricos de 'internet'


# A continuacion procedemos a revisar los posibles errores y/o anomalías de cada variable de 'internet'.

# In[23]:


internet['InternetService'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en servicio de internet.


# In[24]:


internet['OnlineSecurity'].sort_values(ascending=True).unique() # Verificamos posibles anomalías.


# In[25]:


internet['OnlineBackup'].sort_values(ascending=True).unique() # Verificamos posibles anomalías.


# In[26]:


internet['DeviceProtection'].sort_values(ascending=True).unique() # Verificamos posibles anomalías.


# In[27]:


internet['TechSupport'].sort_values(ascending=True).unique() # Verificamos posibles anomalías.


# In[28]:


internet['StreamingTV'].sort_values(ascending=True).unique() # Verificamos posibles anomalías.


# In[29]:


internet['StreamingMovies'].sort_values(ascending=True).unique() # Verificamos posibles anomalías.


# In[30]:


phone.info() #Verificamos los campos a nivel general de 'phone'


# Glosario dataset 'phone': 
# 
# * customerID - ID del cliente.
# * MultipleLines - Indicador de si el cliente tiene múltiples líneas telefónicas (puede ser relevante para entender la complejidad del servicio contratado).

# In[31]:


phone.head()  #Vemos una muestra


# In[32]:


#Verificamos duplicidad
duplicados4 = phone.duplicated()
cantidad_duplicados4 = duplicados4.sum()
cantidad_duplicados4 # Finalmente, comprobamos el número de filas duplicadas de 'phone'


# A continuacion procedemos a revisar los posibles errores y/o anomalías de cada variable de 'phone'.

# In[33]:


phone['MultipleLines'].sort_values(ascending=True).unique() # Verificamos posibles anomalías en los datos.


# Para efectos de analizar las variables en su totalidad en funcion de la variable objetivo es necesario unificar los datasets a continuacion:

# In[34]:


# Unir los datasets contract y personal
data = contract.merge(personal, on='customerID', how='left')

# Unir el dataset phone
data = data.merge(phone[['customerID', 'MultipleLines']], on='customerID', how='left')
# Unir el dataset internet
data = data.merge(internet, on='customerID', how='left')

data.info()#Verificamos la unificacion


# <div class="alert alert-block alert-success">
#     <b>Comentarios del Revisor</b> <a class="tocSkip"></a><br>
# Correcto, info(), head()  son herramientas esceneciales que nos ayudaran a hacer un análisis exploratorio inicial. Opcionalmente podrías siempre incluir describe() para tener mejor idea de los valores que toman tus varibales. Continúa con el buen trabajo! </div>

# **Análisis de los datasets**
# 
# 1) Los datasets por separado no tiene valores ausentes, pero al unirlos en un solo dataset se generan valores ausentes que se deben tratar.
# 2) No hay duplicidad de datos en los datasets.
# 3) Los datasets no tienen valores anómalos.
# 4) Los tipos de datos son correctos para cada variable en todos los datasets, excepto aquellos que tienen variables de fecha. Debiese cambiar de 'object' a 'datetime'.
# 5) Los nombres de las columnas es recomendable que estén en minuscula.
# 6) Se recomienda crear una nueva columna para calcular el tiempo de vigencia del contrato.
# 

# ### Preprocesamiento de datos

# Procederemos a corregir los datos según lo analizado anteriormente.

# In[35]:


# Definimos funcion para corregir texto de variables
def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


# In[36]:


# Renombrar las columnas del DataFrame usando la función to_snake_case
data.rename(columns=lambda x: to_snake_case(x), inplace=True)
# Verificamos cambios
data.info()


# Revisando los valores ausentes podemos observar que, por ejemplo, para un 'customer_id' en donde no se tenga información de por ejemplo 'online_security' quiere decir que ese cliente no tiene esa información porque no tiene ese servicio (recordar que se hizo un cruce de datasets).
# 
# Por lo tanto rellenaremos esos valores ausentes con la palabra 'No'.

# In[37]:


for item in ['multiple_lines','internet_service','online_security','online_backup',
             'device_protection','tech_support','streaming_tv','streaming_movies']:
    data[item] = data[item].fillna('No')


# In[38]:


# Procedemos a corregir los tipos de datos de fechas.
data['begin_date'] = pd.to_datetime(data['begin_date'],format='%Y-%m-%d')
data['end_date'] = pd.to_datetime(data['end_date'],format='%Y-%m-%d %H:%M:%S', errors='coerce')


# Por efectos de **multicolinealidad**, eliminamos el valor 'total_charges' ya que 'monthly_charges' refleja mejor la carga financiera recurrente del cliente y puede ser más dinámico para detectar cambios en el comportamiento del cliente. Además, se decidió no incorporar una columna de duración de contratos por el mismo efecto de multicolinealidad con la variable objetivo 'contract_status'.

# In[39]:


# Eliminar columna 'total_charges' para que no afecte el modelo.
data = data.drop(columns=['total_charges'])


# In[40]:


data.info()


# ### Distribución de datos - Gráficos

# In[41]:


# Estilos de Seaborn
sns.set(style="whitegrid")

# Boxplots para detectar outliers en variables numéricas
for column in data.select_dtypes(include=['float64']).columns:
    plt.figure(figsize=(8, 4))  # Tamaño de la figura más pequeño
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}', fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


# In[42]:

# Filtramos solo las columnas numéricas
numeric_data = data.select_dtypes(include=['number'])

# Matriz de correlación (Variables numéricas vs variable objetivo)
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[43]:


# Definimos la función para visualizar histogramas de variables categóricas
def visualize_histograms(df, columns, title):
    categorical_columns = [col for col in columns if df[col].dtype == 'object']
    n = len(categorical_columns)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12, 5 * rows))
    axes = axes.flatten()
    for i, column in enumerate(categorical_columns):
        axes[i].hist(df[column].dropna(), bins=30, alpha=0.7, edgecolor='black')  # Añadir borde negro a las barras
        axes[i].set_title(f'Histograma de {column}')
        
        # Ajustamos los ticks del eje x para mejor visibilidad
        axes[i].tick_params(axis='x', labelsize=10)  # Ajusta el tamaño de las etiquetas del eje x
        axes[i].tick_params(axis='y', labelsize=10)  # Ajusta el tamaño de las etiquetas del eje y
    
        for tick in axes[i].get_xticklabels():
            tick.set_rotation(90)
        
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Definimos la función para visualizar gráficos de línea de variables numéricas
def visualize_line_plots(df, columns, title):
    numeric_columns = [col for col in columns if df[col].dtype == 'float64']
    n = len(numeric_columns)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12, 5 * rows))
    axes = axes.flatten()
    for i, column in enumerate(numeric_columns):
        sorted_data = df[column].dropna().sort_values().reset_index(drop=True)
        axes[i].plot(sorted_data, alpha=0.7)
        axes[i].set_title(f'Gráfico de Línea de {column}')
        axes[i].set_xlim(left=0, right=len(sorted_data))
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)
        for tick in axes[i].get_xticklabels():
            tick.set_rotation(90)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
# Definimos la función para visualizar gráficos de torta de variables de tipo int
def visualize_pie_charts(df, columns, title):
    int_columns = [col for col in columns if df[col].dtype == 'int64']
    n = len(int_columns)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12, 5 * rows))
    axes = axes.flatten()
    for i, column in enumerate(int_columns):
        data = df[column].value_counts()
        axes[i].pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
        axes[i].set_title(f'Gráfico de Torta de {column}')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()    

# Función para crear tablas de contingencia y gráficos de calor
def visualize_contingency_heatmap(df, categorical_columns, target_column, title):
    for column in categorical_columns:
        contingency_table = pd.crosstab(df[column], df[target_column])
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f'Tabla de Contingencia y Gráfico de Calor de {column} vs {target_column}')
        plt.show() 


# In[44]:


# Dataset completo 'data'
columns_data = data.drop(columns=['customer_id', 'begin_date', 'end_date']).columns

# Visualizar histogramas de variables categóricas
visualize_histograms(data, columns_data, 'Histogramas de variables categóricas del dataset data')

# Visualizar gráficos de línea de variables numéricas
visualize_line_plots(data, columns_data, 'Gráfico de Línea de variables numéricas del dataset data')

# Visualizar gráficos de torta de variables de tipo int
visualize_pie_charts(data, columns_data, 'Gráficos de Torta de variables enteras del dataset data')

# Excluyendo las columnas 'MonthlyChanges' y 'ContractStatus' para la visualización del mapa de calor contra la variable objetivo.
heatmap_columns = [col for col in columns_data if col not in ['monthly_charges', 'contract_status']]
visualize_contingency_heatmap(data, heatmap_columns, 'contract_status', 'Tablas de Contingencia del dataset data')


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Excelente trabajo con los gráficos en esta sección. Sin embargo, te recomiendo ser más específico al visualizar los datos, diferenciando claramente entre variables categóricas y numéricas. Recuerda que el histograma no es la mejor opción para visualizar variables categóricas; para estas, podrías usar gráficos de barras o diagramas de pastel, que proporcionan una representación más clara y precisa de la distribución de categorías.
# </div>

# 
# **Respecto a la distribución de los datos en del dataset completo:**
# * En la variable 'Type', 'Month-to-month' tiene mayor frecuencia que las demás, por lo que podría ocasionar distorsion en el modelo.
# * Las variables PaperlessBilling y  PaymentMethod tienen una distribucion relativamente homogenea, excepto el pago por medio de cheque electrónico (electronic check), el cual tiene mayor preferencia.
# * En el cargo mensual, se puede apreciar un pago mucho más frecuente entre los 0 a 20 dolares mensuales. Esto puede marcar diferencia en el modelo predictivo.
# * Si vemos los cargos totales, la distribución pasa a ser homogenea en todos sus rangos.
# * Podemos ver en la variable objetivo 'ContractStatus' que el valor 1 tiene mas frecuencia (5000) con respecto al valor 0 (poco menos de 2000 casos), es decir, la mayoría de contratos no han sido cancelados.
# * En relacion al genero, no se ven una tendencia clara, lo cual puede ser irrelevante para el modelo y respecto a si son mayores de edad (SeniorCitizen) se puede ver que una minoría lo es..
# * Respecto a si tienen pareja o no, este no hay una tendencia clara, pero respecto a si tiene dependientes en su familia sólo 2/7 lo tienen.
# * Respecto al dataset 'internet' se ve que hubo más casos de usuarios que usaron megas más bajos.
# 
# **Respecto a la correlacion con la variable objetivo (categorica) sobre el término de contrato:**
# 
# * El término de contrato se da más en los contratos mensuales, no de largo plazo.
# * Si hay facturacion electrónica hay mas casos de termino de contrato.
# * El término se da más por el metodo de pago 'electronic check'.
# * Se ve una leve tendencia al termino de contrato si no son 'senior citizen'.
# * Los que no tienen dependientes ni pareja suelen terminar sus contratos.
# * No se ve tendencia si el servicio tiene múltiples lineas.

# **PREGUNTAS ACLARATORIAS**

# 1) El enunciado del proyecto dice que la información de los contratos cuenta a partir del '2020-02-01' pero analizando los datos solo 11 contratos inician en esa fecha. ¿Hay un error en el dataset 'contract' o el enunciado quiere decir otra cosa?
# 
# 2) El dataset 'internet' no tiene el 'customerID' por lo que no puedo enlazar su información con los demas datasets.
# 
# 3) ¿Será oportuno calcular la duración de los contratos (valor duracion en meses o dias) y el tiempo restante del contrato (fecha inicio hasta 'no')? De ser así, ¿como podria dejar esa columna de duracion de contrato, es decir, con valores numéricos y una categoria de 'vigente' en una columna? 
# 
# 4) ¿Que quiere decir exactamente la variable 'session_date' del dataset 'internet'? ¿Debiese además haber una variable de tiempo para poder obtener el uso promedio de 'mb' por mes?
# 
# 5) El enunciado del proyecto habla de que Interconnect proporciona otros tipos de servicios, como: Seguridad en Internet y un bloqueador de sitios web maliciosos, una línea de soporte técnico, almacenamiento de archivos en la nube y backup de datos, y finalmente, Streaming de TV y directorio de películas, pero estos servicios no aparecen en los dataset entregados. ¿efectivamente falta algun otro dataset?.
# 
# 6) ¿Para el modelo debiese trabajar con el cobro mensual o cobro total? ¿Debería usar solo uno, no ambos, cierto?
# 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# 
# Has planteado preguntas importantes que necesitan clarificación para avanzar correctamente en el proyecto:
# 
# 1) **Sobre la información de contratos a partir del '2020-02-01'**: Hay un error en el enunciado. 
# 
# 2) **Falta de 'customerID' en el dataset 'internet'**: Es un problema significativo, ya que impide enlazar esta información con otros datasets. Te aconsejo revisar los datasets de la siguiente carpeta, pues creo son diferentes y eso es el problema: 
# <code>
# contract = pd.read_csv('/datasets/final_provider/contract.csv')
# internet = pd.read_csv('/datasets/final_provider/internet.csv')
# </code>    
# 
# 3) **Cálculo de la duración de los contratos**: Sí, es muy oportuno calcular la duración de los contratos y el tiempo restante. Podrías crear una columna de duración en días o meses y otra columna para indicar si el contrato está vigente ('vigente') o terminado. Tal como lo has planteado. 
# 
# 4) **Clarificación de 'session_date' en el dataset 'internet'**: Sí representa fechas de uso de servicio. 
# 
# 5) **Servicios adicionales de Interconnect**: No hay data sets adicionales. Aconsejo que no revisemos servicios adicionales. 
#     
# 
# 6) **Cobro mensual vs cobro total**: Para el modelo predictivo, generalmente es mejor usar una sola variable para evitar multicolinealidad. Entre cobro mensual y cobro total, elegiría el cobro mensual, ya que refleja mejor la carga financiera recurrente del cliente y puede ser más dinámico para detectar cambios en el comportamiento del cliente.
# 
# </div>

# **PASOS A SEGUIR PARA RESOLVER LA TAREA**

# 1) Una vez realizado el análisis exploratorio de datos, debiese pasar a la etapa de 'Preparación de datos' según las conclusiones obtenidas de ese análisis. Esto incluye: ajuste de nombres en minusculas, tipo de datos de fechas, unificar datasets(debe incluir la informacion faltante de los servicios extras ademas de telefonia e internet), conversión de variables categóricas en numéricas para realizar el modelo, normalizar las variables numéricas, examinar equilibrio de clases.
# 
# 2) Dado que el objetivo es predecir una variable binaria (cancelación o no cancelación), se usarán modelos de clasificación supervisada, donde inicialmente se entrenará y evaluará el modelo de regresión lineal para tener un análisis base.
# 
# 3) Luego, seleccionamos el modelo más optimo basado en las métricas de evaluación como precision, recall, f1-score y AUC-ROC e hiperparámetros a través de métodos iterativos. Modelos pueden ser: Árboles de Decisión y Random Forest, o modelos más complejos con alto rendimiento como Gradient Boosting Machines (GBM) y XGBoost.
# 

# ## Examinación de equilibrio de clases

# Antes de examinar las clases, debemos preparar el dataset con la mayor cantidad de valores numéricos, por ejemplo, transformaremos las variables con datos 'Yes' en 1 y 'No' en 0.

# In[45]:


data.info()


# In[46]:


# Función para transformar las variables con Yes y No en 1 y 0.
def transform_yes_no_to_binary(df, columns):
    for column in columns:
        df[column] = df[column].map({'Yes': 1, 'No': 0})
    return df


# Lista de columnas a transformar
columns_to_transform = ['paperless_billing', 'partner', 'dependents','multiple_lines','online_security','online_backup',
                        'device_protection','tech_support','streaming_tv','streaming_movies'] # columnas categoricas

#  Se llama a la función para transformar el dataset 'data'
data = transform_yes_no_to_binary(data, columns_to_transform)

# Verificamos la transformación
print(data[columns_to_transform].head())


# In[49]:


# Convertimos las variables categóricas del dataset en representaciones numéricas usando one-hot encoding
data_balanced = pd.get_dummies(data, columns=['type','payment_method','gender','internet_service'])


# In[54]:


data_balanced.head()


# In[55]:


# Recordamos el equilibrio de clases de la variable objetivo.
clases_equilibradas = data_balanced['contract_status'].value_counts(normalize=True)
clases_equilibradas


# Esto significa que aproximadamente el 73.46% de las observaciones en el conjunto de datos pertenecen a la clase 1 (contratos vigentes), mientras que aproximadamente el 26.54% pertenecen a la clase 0 (contratos dados de baja).

# En este caso, hay un desequilibrio significativo, ya que la clase 1 es mucho más común que la clase 0 (contratos dados de baja). Tendremos en cuenta este desequilibrio al entrenar y evaluar los modelos, ya que puede afectar la capacidad del modelo para aprender correctamente patrones en la clase minoritaria y puede sesgar las predicciones hacia la clase mayoritaria.

# Eliminamos las columnas que no usaremos en el modelo por concepto de MULTICOLINEALIDAD.
data_balanced_modelo = data_balanced.drop(columns=['customer_id', 'begin_date', 'end_date'])

# ## Preparación del modelo
# ### Modelo Equilibrado (submuestreo de clase mayoritaria)
# #### División de los conjuntos

# Separación de características y target
x = data_balanced_modelo.drop('contract_status', axis=1)
y = data_balanced_modelo['contract_status']

# Dividimos el conjunto de datos en entrenamiento + validación y prueba
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Dividimos el conjunto de entrenamiento + validación en conjunto de entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# Revisamos la distribución de las clases
print("Distribución en el conjunto de entrenamiento:", y_train.value_counts())
print("Distribución en el conjunto de validación:", y_val.value_counts())
print("Distribución en el conjunto de prueba:", y_test.value_counts())

# #### Equilibrio por submuestreo de clase mayoritaria

# Combinamos x_train e y_train en un solo DataFrame para facilitar el muestreo
train_data = pd.concat([x_train, y_train], axis=1)

# Separamos la clase mayoritaria y la minoritaria
df_majority = train_data[train_data.contract_status == 1]
df_minority = train_data[train_data.contract_status == 0]

# Submuestreo de la clase mayoritaria
df_majority_downsampled = resample(df_majority, 
                                   replace=False,    # muestra sin reemplazo
                                   n_samples=len(df_minority), # para igualar el número de la clase minoritaria
                                   random_state=42)  # para la reproducibilidad

# Combinamos clases minoritaria y mayoritaria
train_data_balanced = pd.concat([df_majority_downsampled, df_minority])

# Separamos de nuevo características y target
x_train_balanced = train_data_balanced.drop('contract_status', axis=1)
y_train_balanced = train_data_balanced['contract_status']

# Revisamos la distribución de las clases en el conjunto de entrenamiento equilibrado
print("Distribución en el conjunto de entrenamiento equilibrado:", y_train_balanced.value_counts())

# #### Iteración de modelos

# Definimos modelos candidatos
models = {
    'RandomForest': RandomForestClassifier(random_state=12345),
    'DecisionTree': DecisionTreeClassifier(random_state=12345),
    'LogisticRegression': LogisticRegression(random_state=12345),
    'LightGBM': LGBMClassifier(random_state=12345),
    'CatBoost': CatBoostClassifier(random_state=12345, verbose=0),
    'XGBoost': XGBClassifier(random_state=12345, use_label_encoder=False, eval_metric='logloss'),
    'GradientBoosting': LGBMClassifier(boosting_type='gbdt', random_state=12345)  # Agregamos Gradient Boosting
}


# Definimos hiperparámetros con rangos para iteración según aplique a cada modelo.
param_grid = {
    'RandomForest': {'n_estimators': range(10, 101, 10), 'max_depth': range(1, 21)},
    'DecisionTree': {'max_depth': range(1, 21)},
    'LogisticRegression': {},  # No hay hiperparámetros para LogisticRegression
    'LightGBM': {'n_estimators': range(50, 201, 50), 'max_depth': range(1, 21)},
    'CatBoost': {'depth': range(1, 11), 'iterations': range(50, 201, 50)},
    'XGBoost': {'n_estimators': range(50, 201, 50), 'max_depth': range(1, 21)},
    'GradientBoosting': {'n_estimators': range(50, 201, 50), 'max_depth': range(1, 21)}  # Hiperparámetros para Gradient Boosting
}

# Definimos parámetros para iteración de entrenamiento y ajuste de modelos
best_model_1 = None
best_f1_score_1 = 0

for name, model in models.items():
    params = param_grid[name]
    best_params = None
    best_model_instance = None
    best_f1_score_instance = 0
    
    for n_estimators in params.get('n_estimators', [None]):
        for max_depth in params.get('max_depth', [None]):
            if name == 'RandomForest':
                model_instance = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=12345)
            elif name == 'DecisionTree':
                model_instance = DecisionTreeClassifier(max_depth=max_depth, random_state=12345)
            elif name == 'LightGBM':
                model_instance = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=12345)
            elif name == 'CatBoost':
                model_instance = CatBoostClassifier(depth=max_depth, iterations=n_estimators, random_state=12345, verbose=0)
            elif name == 'XGBoost':
                model_instance = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=12345, use_label_encoder=False, eval_metric='logloss')
            elif name == 'GradientBoosting':  # Implementación del Gradient Boosting
                model_instance = LGBMClassifier(boosting_type='gbdt', n_estimators=n_estimators, max_depth=max_depth, random_state=12345)
            else:
                model_instance = LogisticRegression(random_state=12345)
            
            # Entrenamos con los datos balanceados
            model_instance.fit(x_train_balanced, y_train_balanced)
            
            # Evaluamos en el conjunto de validación
            y_val_pred = model_instance.predict(x_val)
            f1 = f1_score(y_val, y_val_pred)
            
            if f1 > best_f1_score_instance:
                best_f1_score_instance = f1
                best_model_instance = model_instance
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    
    print(f"Mejor F1-score para {name}: {best_f1_score_instance} con hiperparámetros: {best_params}")
    
    if best_f1_score_instance > best_f1_score_1:
        best_model_1 = best_model_instance
        best_f1_score_1 = best_f1_score_instance

# Evaluación del mejor modelo en el conjunto de prueba
y_test_pred = best_model_1.predict(x_test)
test_f1_1 = f1_score(y_test, y_test_pred)
print(f"Mejor modelo en conjunto de prueba: {best_model_1} con F1-score = {test_f1_1}")

# ### Modelo Equilibrado (sobremuestreo clase minoritaria)
# #### División de los conjuntos

# Separación de características y target
a = data_balanced_modelo.drop('contract_status', axis=1)
b = data_balanced_modelo['contract_status']

# Dividimos el conjunto de datos en entrenamiento + validación y prueba
a_train_val, a_test, b_train_val, b_test = train_test_split(a, b, test_size=0.2, random_state=42, stratify=b)

# Dividimos el conjunto de entrenamiento + validación en conjunto de entrenamiento y validación
a_train, a_val, b_train, b_val = train_test_split(a_train_val, b_train_val, test_size=0.25, random_state=42, stratify=b_train_val)

# Revisamos la distribución de las clases
print("Distribución en el conjunto de entrenamiento:", b_train.value_counts())
print("Distribución en el conjunto de validación:", b_val.value_counts())
print("Distribución en el conjunto de prueba:", b_test.value_counts())

# #### Equilibrio por sobremuestreo de clase minoritaria

# Combinamos a_train e b_train en un solo DataFrame para facilitar el muestreo
train_data = pd.concat([a_train, b_train], axis=1)

# Separamos la clase mayoritaria y la minoritaria
df_majority = train_data[train_data.contract_status == 1]
df_minority = train_data[train_data.contract_status == 0]


# Sobremuestreo de la clase minoritaria
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # muestra con reemplazo
                                 n_samples=len(df_majority), # para igualar el número de la clase mayoritaria
                                 random_state=42)  # para la reproducibilidad


# Combinamos clases minoritaria y mayoritaria
train_data_balanced = pd.concat([df_majority_downsampled, df_minority])

# Separamos de nuevo características y target
a_train_balanced = train_data_balanced.drop('contract_status', axis=1)
b_train_balanced = train_data_balanced['contract_status']

# Revisamos la distribución de las clases en el conjunto de entrenamiento equilibrado
print("Distribución en el conjunto de entrenamiento equilibrado:", b_train_balanced.value_counts())

# #### Iteración de modelos
# Definimos modelos candidatos
models = {
    'RandomForest': RandomForestClassifier(random_state=12345),
    'DecisionTree': DecisionTreeClassifier(random_state=12345),
    'LogisticRegression': LogisticRegression(random_state=12345),
    'LightGBM': LGBMClassifier(random_state=12345),
    'CatBoost': CatBoostClassifier(random_state=12345, verbose=0),
    'XGBoost': XGBClassifier(random_state=12345, use_label_encoder=False, eval_metric='logloss'),
    'GradientBoosting': LGBMClassifier(boosting_type='gbdt', random_state=12345)  # Agregamos Gradient Boosting
}


# Definimos hiperparámetros con rangos para iteración según aplique a cada modelo.
param_grid = {
    'RandomForest': {'n_estimators': range(10, 101, 10), 'max_depth': range(1, 21)},
    'DecisionTree': {'max_depth': range(1, 21)},
    'LogisticRegression': {},  # No hay hiperparámetros para LogisticRegression
    'LightGBM': {'n_estimators': range(50, 201, 50), 'max_depth': range(1, 21)},
    'CatBoost': {'depth': range(1, 11), 'iterations': range(50, 201, 50)},
    'XGBoost': {'n_estimators': range(50, 201, 50), 'max_depth': range(1, 21)},
    'GradientBoosting': {'n_estimators': range(50, 201, 50), 'max_depth': range(1, 21)}  # Hiperparámetros para Gradient Boosting
}

# Definimos parámetros para iteración de entrenamiento y ajuste de modelos
best_model_2 = None
best_f1_score_2 = 0

for name, model in models.items():
    params = param_grid[name]
    best_params = None
    best_model_instance = None
    best_f1_score_instance = 0
    
    for n_estimators in params.get('n_estimators', [None]):
        for max_depth in params.get('max_depth', [None]):
            if name == 'RandomForest':
                model_instance = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=12345)
            elif name == 'DecisionTree':
                model_instance = DecisionTreeClassifier(max_depth=max_depth, random_state=12345)
            elif name == 'LightGBM':
                model_instance = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=12345)
            elif name == 'CatBoost':
                model_instance = CatBoostClassifier(depth=max_depth, iterations=n_estimators, random_state=12345, verbose=0)
            elif name == 'XGBoost':
                model_instance = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=12345, use_label_encoder=False, eval_metric='logloss')
            elif name == 'GradientBoosting':  # Implementación del Gradient Boosting
                model_instance = LGBMClassifier(boosting_type='gbdt', n_estimators=n_estimators, max_depth=max_depth, random_state=12345)
            else:
                model_instance = LogisticRegression(random_state=12345)
            
            # Entrenamos con los datos balanceados
            model_instance.fit(a_train_balanced, b_train_balanced)
            
            # Evaluamos en el conjunto de validación
            b_val_pred = model_instance.predict(a_val)
            f1 = f1_score(b_val, b_val_pred)
            
            if f1 > best_f1_score_instance:
                best_f1_score_instance = f1
                best_model_instance = model_instance
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    
    print(f"Mejor F1-score para {name}: {best_f1_score_instance} con hiperparámetros: {best_params}")
    
    if best_f1_score_instance > best_f1_score_2:
        best_model_2 = best_model_instance
        best_f1_score_2 = best_f1_score_instance

# Evaluación del mejor modelo en el conjunto de prueba
b_test_pred = best_model_2.predict(a_test)
test_f1_2 = f1_score(b_test, b_test_pred)
print(f"Mejor modelo en conjunto de prueba: {best_model_2} con F1-score = {test_f1_2}")

# Evaluamos el mejor modelo en conjunto de prueba
b_pred_test = best_model_2.predict(a_test)
# Evaluamos F1 en el conjunto de prueba
f1_test_2 = f1_score(b_test, b_pred_test)
print("Mejor modelo (basado en F1-score en conjunto de validación) en conjunto de prueba:")
print(classification_report(b_test, b_pred_test))
print(f'F1 en conjunto de prueba: {f1_test_2}')