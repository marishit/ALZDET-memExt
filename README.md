# ALZDET-memExt
Sí, una memoria externa de 500GB debería ser suficiente para manejar los datos de OASIS-1, incluyendo tanto los datos demográficos y clínicos como las imágenes sin procesar y los datos FreeSurfer. A continuación, se detalla cómo proceder con esta configuración:

### Estrategia Detallada

1. **Preparación de la Memoria Externa**

    - **Conecta la memoria externa** a tu laptop.
    - **Crea una estructura de directorios** en la memoria externa para organizar tus datos y entorno de trabajo:

    ```bash
    /path/to/external_drive/
    ├── oasis_data/
    │   ├── raw/
    │   ├── freesurfer/
    │   └── processed/
    ├── my_django_project/
    │   ├── venv/
    │   ├── myproject/
    │   └── myapp/
    ```

### Paso 1: Crear el Entorno Virtual y Configurar Django en la Memoria Externa

#### a. Crear el Entorno Virtual en la Memoria Externa

1. **Navega a la carpeta donde deseas crear el proyecto en la memoria externa**:

    ```bash
    cd /path/to/external_drive/my_django_project
    ```

2. **Crea el entorno virtual en la memoria externa**:

    ```bash
    python -m venv venv
    ```

3. **Activa el entorno virtual**:

    - En Windows:

      ```bash
      /path/to/external_drive/my_django_project/venv/Scripts/activate
      ```

    - En macOS/Linux:

      ```bash
      source /path/to/external_drive/my_django_project/venv/bin/activate
      ```

4. **Instala Django y otras dependencias**:

    ```bash
    pip install django pandas nibabel scikit-learn numpy
    ```

#### b. Configurar Django en la Memoria Externa

1. **Crear un proyecto Django en la memoria externa**:

    ```bash
    django-admin startproject myproject
    cd myproject
    ```

2. **Crear una aplicación dentro del proyecto**:

    ```bash
    python manage.py startapp myapp
    ```

3. **Configurar la base de datos y las aplicaciones en `settings.py`**:

    Edita `myproject/settings.py` para incluir tu aplicación en `INSTALLED_APPS` y configura la base de datos si es necesario.

4. **Realizar las migraciones y arrancar el servidor de desarrollo**:

    ```bash
    python manage.py migrate
    python manage.py runserver
    ```

### Paso 2: Descarga y Extracción de Datos en la Memoria Externa

1. **Descarga los archivos necesarios desde la página de OASIS-1 directamente a la memoria externa**:

    - Datos demográficos y clínicos
    - Imágenes sin procesar (`oasis_cross-sectional_disc1.tar.gz`, `oasis_cross-sectional_disc2.tar.gz`, etc.)
    - Datos FreeSurfer (si los necesitas)

2. **Extrae los archivos en la memoria externa**:

    ```bash
    cd /path/to/external_drive/oasis_data/raw
    tar -xzvf oasis_cross-sectional_disc1.tar.gz
    tar -xzvf oasis_cross-sectional_disc2.tar.gz
    ```

### Paso 3: Preprocesamiento de Datos

1. **Cargar datos demográficos y clínicos**:

    ```python
    import pandas as pd

    # Ruta al archivo CSV en la memoria externa
    csv_file_path = '/path/to/external_drive/oasis_data/raw/oasis_demographics_and_clinical_data.csv'
    clinical_data = pd.read_csv(csv_file_path)

    # Mostrar información básica
    print(clinical_data.head())
    ```

2. **Procesar imágenes cerebrales**:

    ```python
    import nibabel as nib
    import numpy as np

    def load_and_process_image(image_path):
        img = nib.load(image_path)
        img_data = img.get_fdata()
        # Normalización de la imagen
        img_data = (img_data - np.mean(img_data)) / np.std(img_data)
        return img_data

    # Ejemplo de carga de una imagen
    image_path = '/path/to/external_drive/oasis_data/raw/subject1.nii.gz'
    img_data = load_and_process_image(image_path)

    # Mostrar forma de los datos de imagen
    print(img_data.shape)
    ```

3. **Guardar características extraídas**:

    ```python
    import os

    def save_features(patient_id, features, output_dir='/path/to/external_drive/oasis_data/processed'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        feature_path = os.path.join(output_dir, f'{patient_id}_features.npy')
        np.save(feature_path, features)

    # Ejemplo de guardado de características
    patient_id = 'subject1'
    save_features(patient_id, img_data)
    ```

### Paso 4: Entrenamiento del Modelo

1. **Cargar datos preprocesados y entrenar el modelo**:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score

    # Preprocesamiento de datos clínicos
    X = clinical_data.drop(columns=['diagnosis'])
    y = clinical_data['diagnosis']

    # Normalización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reducción de dimensionalidad
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    # Entrenamiento del modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluación del modelo
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f'AUC-ROC: {roc_auc_score(y_test, y_pred)}')
    ```

### Conclusión

Una memoria externa de 500GB debería ser suficiente para manejar los datos de OASIS-1 y el entorno de desarrollo. Al almacenar todo el proyecto y los datos en la memoria externa, podrás gestionar mejor el espacio en tu disco interno y mantener una estructura de proyecto bien organizada. Esta estrategia te permitirá trabajar de manera eficiente y eficaz con los datos de OASIS-1 en tu laptop.
