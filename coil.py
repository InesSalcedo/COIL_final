import pandas as pd

# Ruta del archivo de Excel
excel_file = '/workspaces/COIL_final/2023_Accidentalidad.xlsx'

# Cargar los datos del archivo de Excel en un DataFrame
df = pd.read_excel(excel_file)

# Realizar un análisis estadístico básico para la variable "hombre"
stats_hombre = df['hombre'].describe()

# Realizar un análisis estadístico básico para la variable "mujer"
stats_mujer = df['mujer'].describe()

# Comparar las variables "hombre" y "mujer"
# Calculate the difference between stats_hombre and stats_mujer
comparison = stats_hombre - stats_mujer

# Print the comparison results
print("Comparison between 'hombre' and 'mujer' variables:")
print(comparison)

# Calculate the correlation matrix between 'hombre' and 'mujer' variables
correlation_matrix = df[['hombre', 'mujer']].corr()

# Print the correlation table
print("\nCorrelation table between 'hombre' and 'mujer' variables:")
print(correlation_matrix)






