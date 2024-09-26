import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, KeepTogether, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Función para generar tablas
def generate_table(dataset_name, distribution, k_values, model, train_data, target):
    table = {'Dataset': [], 'Distribución': [], 'No. Pliegues (k)': [], 'Pliegue': [], 'Accuracy': []}
    
    for k in k_values:
        scores = cross_val_score(model, train_data, target, cv=k)
        
        for i, score in enumerate(scores):
            table['Dataset'].append(dataset_name)
            table['Distribución'].append(distribution)
            table['No. Pliegues (k)'].append(k)
            table['Pliegue'].append(i + 1)
            table['Accuracy'].append(round(score, 4))  # Reducir a 4 decimales
        
        table['Dataset'].append(dataset_name)
        table['Distribución'].append(distribution)
        table['No. Pliegues (k)'].append(k)
        table['Pliegue'].append('Promedio')
        table['Accuracy'].append(round(scores.mean(), 4))  # Reducir a 4 decimales
    
    return pd.DataFrame(table)

# Función para generar tablas de clasificación
def generate_classification_table(dataset_name, model, train_data, test_data, target):
    model.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
    predictions = model.predict(test_data.iloc[:, :-1])
    
    # Obtener el informe de clasificación como DataFrame
    report_df = pd.DataFrame(classification_report(target, predictions, output_dict=True)).transpose()
    
    # Reducir la cantidad de decimales en las columnas numéricas
    for col in report_df.columns:
        if report_df[col].dtype == 'float64':
            report_df[col] = round(report_df[col], 4)
    
    # Agregar columnas adicionales para el dataset y el modelo
    report_df['Dataset'] = dataset_name
    report_df['Model'] = model.__class__.__name__
    
    return report_df

# Función para agregar tabla de clasificación al PDF
def add_classification_table_to_pdf(table, title):
    styles = getSampleStyleSheet()
    
    # Reducir la cantidad de decimales en las celdas de la tabla
    data = [table.columns.values.astype(str)] + table.round(4).values.tolist()

    pdf_table = Table(data, style=[
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),  # Ajustar el espacio en la parte inferior de la primera fila
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])

    # Agregar la tabla al contenido del documento PDF
    return [Paragraph(title, styles['Heading2']), pdf_table]  # Ajustar el estilo del título según tus preferencias

# Función para agregar una tabla de resumen al PDF
def add_summary_table_to_pdf(data, title):
    styles = getSampleStyleSheet()

    # Ajustar la cantidad de decimales en los valores numéricos
    data = data.round(4).values.tolist()

    pdf_table = Table(data, style=[
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),  # Ajustar el espacio en la parte inferior de la primera fila
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])

    # Agregar la tabla al contenido del documento PDF
    return [Paragraph(title, styles['Heading2']), pdf_table]  # Ajustar el estilo del título según tus preferencias

# Función para generar y guardar informe de matriz de confusión
def generate_confusion_matrix_report(dataset_name, model, train_data, test_data, target):
    try:
        model.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
    except ValueError as e:
        if "seen at fit time, yet now missing" in str(e):
            model.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
        else:
            raise e

    predictions = model.predict(test_data.iloc[:, :-1])

    cm = confusion_matrix(target, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target.unique())

    plt.figure(figsize=(8, 8))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'{dataset_name} - Confusion Matrix')
    
    # Crear un nombre de archivo único usando el nombre del conjunto de datos y el tipo de modelo
    confusion_matrix_filename = f'{dataset_name}_{model.__class__.__name__.lower()}_confusion_matrix.png'
    
    plt.savefig(confusion_matrix_filename)

    # Guardar informe de clasificación
    report_df = pd.DataFrame(classification_report(target, predictions, output_dict=True)).transpose()
    report_df['Dataset'] = dataset_name
    report_df['Model'] = model.__class__.__name__
    report_df.round(4).to_csv(f'{dataset_name}_{model.__class__.__name__.lower()}_classification_report.csv', index=False)

    return confusion_matrix_filename

# Función para agregar imagen de matriz de confusión al PDF
def add_confusion_matrix_image_to_pdf(image_filename, title):
    return [
        Table([[f'{title}:']], colWidths=[6 * inch], rowHeights=0.4 * inch),
        Image(image_filename, width=5*inch, height=5*inch)
    ]

# Función para agregar tabla al PDF
def add_table_to_pdf(table, title):
    styles = getSampleStyleSheet()
    data = table.round(4).values.tolist()
    col_widths = [1.0 * inch] * len(data[0])  # Ajustar el ancho de las columnas según tus preferencias

    pdf_table = Table([table.columns.values.astype(str)] + data, colWidths=col_widths, rowHeights=0.25 * inch)  # Ajustar la altura de las filas
    pdf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),  # Ajustar el espacio en la parte inferior de la primera fila
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    # Agregar la tabla al contenido del documento PDF
    return [Paragraph(title, styles['Heading2']), pdf_table]  # Ajustar el estilo del título según tus preferencias

# Cargar conjuntos de datos
iris_data = pd.read_csv('iris.csv')
emails_data = pd.read_csv('emails.csv')

# Mezclar datos
iris_data = shuffle(iris_data, random_state=0)
emails_data = shuffle(emails_data, random_state=0)

# Dividir en conjuntos de entrenamiento y prueba para Iris
iris_train, iris_test = train_test_split(iris_data, test_size=0.3, random_state=0)

# Eliminar la primera columna de datos de correo electrónico
emails_data = emails_data.iloc[:, 1:]
emails_train, emails_test = train_test_split(emails_data, test_size=0.3, random_state=0)

# Configuración de validación cruzada
k_values = [3, 5]

# Modelos
gnb = GaussianNB()
mnb = MultinomialNB()

# Generar tablas para iris
iris_normal_table = generate_table('iris', 'Normal', k_values, gnb, iris_train.iloc[:, :-1], iris_train.iloc[:, -1])
iris_multinomial_table = generate_table('iris', 'Multinomial', k_values, mnb, iris_train.iloc[:, :-1], iris_train.iloc[:, -1])

# Generar tablas para correos electrónicos
emails_normal_table = generate_table('emails', 'Normal', k_values, gnb, emails_train.iloc[:, 1:], emails_train.iloc[:, -1])
emails_multinomial_table = generate_table('emails', 'Multinomial', k_values, mnb, emails_train.iloc[:, 1:], emails_train.iloc[:, -1])

# Generar y guardar tablas de clasificación para iris
iris_gnb_classification_table = generate_classification_table('iris', gnb, iris_train, iris_test, iris_test.iloc[:, -1])
iris_mnb_classification_table = generate_classification_table('iris', mnb, iris_train, iris_test, iris_test.iloc[:, -1])

# Generar y guardar tablas de clasificación para correos electrónicos
emails_gnb_classification_table = generate_classification_table('emails', gnb, emails_train, emails_test, emails_test.iloc[:, -1])
emails_mnb_classification_table = generate_classification_table('emails', mnb, emails_train, emails_test, emails_test.iloc[:, -1])

# Crear archivo PDF
pdf_filename = 'output_results.pdf'
doc = SimpleDocTemplate(pdf_filename, pagesize=letter)

# Crear lista para almacenar todos los elementos (tablas e imágenes) del PDF
elements = []

# Agregar tablas a la lista de elementos
elements.extend([
    KeepTogether(add_table_to_pdf(iris_normal_table, 'Iris Normal Table')),
    KeepTogether(add_table_to_pdf(iris_multinomial_table, 'Iris Multinomial Table')),
    KeepTogether(add_table_to_pdf(emails_normal_table, 'Emails Normal Table')),
    KeepTogether(add_table_to_pdf(emails_multinomial_table, 'Emails Multinomial Table')),
    KeepTogether(add_table_to_pdf(iris_gnb_classification_table, 'Iris GNB Classification Table')),
    KeepTogether(add_table_to_pdf(iris_mnb_classification_table, 'Iris MNB Classification Table')),
    KeepTogether(add_table_to_pdf(emails_gnb_classification_table, 'Emails GNB Classification Table')),
    KeepTogether(add_table_to_pdf(emails_mnb_classification_table, 'Emails MNB Classification Table'))
])

# Generar y guardar matrices de confusión e informes para iris
iris_gnb_conf_matrix = generate_confusion_matrix_report('iris', gnb, iris_train, iris_test, iris_test.iloc[:, -1])
iris_mnb_conf_matrix = generate_confusion_matrix_report('iris', mnb, iris_train, iris_test, iris_test.iloc[:, -1])

# Generar y guardar matrices de confusión e informes para correos electrónicos
emails_gnb_conf_matrix = generate_confusion_matrix_report('emails', gnb, emails_train, emails_test, emails_test.iloc[:, -1])
emails_mnb_conf_matrix = generate_confusion_matrix_report('emails', mnb, emails_train, emails_test, emails_test.iloc[:, -1])

# Agregar imágenes de matrices de confusión a la lista de elementos
elements.extend([
    KeepTogether(add_confusion_matrix_image_to_pdf(iris_gnb_conf_matrix, 'Iris GNB Confusion Matrix')),
    KeepTogether(add_confusion_matrix_image_to_pdf(iris_mnb_conf_matrix, 'Iris MNB Confusion Matrix')),
    KeepTogether(add_confusion_matrix_image_to_pdf(emails_gnb_conf_matrix, 'Emails GNB Confusion Matrix')),
    KeepTogether(add_confusion_matrix_image_to_pdf(emails_mnb_conf_matrix, 'Emails MNB Confusion Matrix'))
])

# Generar y guardar tablas de resumen
summary_data = pd.DataFrame({
    'Dataset': ['iris', 'emails'],
    'Distribución': ['Normal', 'Multinomial'],
    'No. Pliegues': [k_values[-1]] * 2,
    'Accuracy': [
        iris_normal_table.loc[iris_normal_table['Pliegue'] == 'Promedio', 'Accuracy'].values[0],
        emails_multinomial_table.loc[emails_multinomial_table['Pliegue'] == 'Promedio', 'Accuracy'].values[0]
    ]
})

# Agregar encabezado a la tabla de resumen
summary_data = pd.DataFrame({
    'Dataset': ['Dataset', 'iris', 'emails'],
    'Distribución': ['Distribución', 'Normal', 'Multinomial'],
    'No. Pliegues': ['No. Pliegues', k_values[-1], k_values[-1]],
    'Accuracy': ['Accuracy'] + [
        iris_normal_table.loc[iris_normal_table['Pliegue'] == 'Promedio', 'Accuracy'].values[0],
        emails_multinomial_table.loc[emails_multinomial_table['Pliegue'] == 'Promedio', 'Accuracy'].values[0]
    ]
})

# Agregar la tabla de resumen a la lista de elementos
elements.append(KeepTogether(add_summary_table_to_pdf(summary_data, 'Summary Table')))

# Construir el PDF con todos los elementos
doc.build(elements)
