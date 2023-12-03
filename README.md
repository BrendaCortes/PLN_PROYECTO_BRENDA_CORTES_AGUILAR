# PLN_PROYECTO_BRENDA_CORTES_AGUILAR

Descripción y objetivos del proyecto.
Objetivos:
•	Aplicar técnicas de procesamiento de lenguaje natural para extraer información de textos.
•	Aplicar conocimientos de ciencia de datos con el procesamiento de lenguaje natural.
•	Realizar predicciones incluso en un dataset que solo contiene datos de tipo texto.

Descripción de la actividad:

Este proyecto consiste en cuatro etapas principales:
1.	EDA.
2.	Creación de un analizador de sentimientos.
3.	Creación de modelos de ML para realizar predicciones.
4.	Presentación del proyecto.

En esta ocasión se utilizarán 1 conjuntos de datos llamado “reviews.csv”, este dataset contiene información sobre las reseñas de personas aleatorias respecto a la adaptación a live-action del anime “One Piece”. 


# 1. ETAPA	EDA.

Paso 1. Importe las librerías necesarias (pandas, numpy, seaborn, nltk, etc...)
Paso 2. Cargue y muestre información del dataset; muestre información estadística de las columnas numéricas.

Podemos observar que el data set contiene 4 columnas, con 878 registros, donde únicamente podemos observar una única columna de tipo numérica llamada 'Rating'.

Paso 3. Identifique los datos nulos: muestre las filas que contienen datos nulos (no se deben tratar aún).

Podemos observar que "Rating" tiene 8 valores nulos. Las otras columnas, "Title", "Review" y "Date", no tienen valores nulos. Por la cantidad de datos, estos datos se podrían eliminar, en caso de que sea necesario aplicarles algún tratamiento.

Paso 4. Muestre la distribución de la columna "Rating", haga un análisis de la distribución.

Podemos observar que la variable'Rating' cuanta con valores que varían entre 1, 2, ..., 10, entonces podemos decir que estamos frente a una variable discreta, por lo que podríamos encontrarnos frente a un problema de clasificación, esta variable representa la calificación que obtuvo el live-action y por la forma de la distribución, podríamos decir que la mayoría de las calificaciones son altas.

Paso 5. Identifique si alguna de las columnas se puede convertir en categórica.

Una vez analizado la cantidad de datos únicos y dado el contexto de las variables:

Title y Review: Estas columnas tienen un número muy grande de valores únicos. Convertirlas a categóricas podría no ser beneficioso, ya que las categorías podrían ser demasiado específicas.

Date: Tiene 40 valores únicos. Esta variable podría ser candidata, porque únicamente se registran variación de dos meses en un mismo año, y varían, dado que se considera el día, podríamos llevar una categorización por mes; sin embargo, debemos de analizar que tan conveniente podría llegar a ser.

Rating: Tiene 10 valores únicos, aunque podemos tratarla como una clasificación, convertir esta columna a categórica podría no ser apropiado porque podemos trabajar con los valores numéricos.


# 2. ETAPA 2 ANÁLISIS DE SENTIMIENTOS.

Paso 1. Muestre las primeras 10 filas del dataset con las columnas "Rating" y "Review", haga un análisis rápido de esa información.

Paso 2. Haga una función que se encargue del pre-procesamiento:
- Genere los tokens.
- Filtre las palabras de parada.
- Obtenga el lema de las palabras y guárdelo en una lista.
- Retorne la lista en forma de una cadena, para ello debe unir los elementos de la lista mediante un espacio

Paso 3. Aplique la función creada para obtener el lema de las columnas "Review" y "Title", guárde el resultado en nuevas columnas dentro del dataframe original (por ejemplo: "ReviewText", "TitleText").

Paso 4. Haga una función para obtener el sentimiento de las palabras, para ello puede utilizar el SentimentIntensityAnalizer() y su función "polarity_scores()". Al final debe retornar el puntaje de sentimiento.

Paso 5. Aplique la función creada para obtener el sentimiento en las columnas creadas en el paso 3, guarde el resultado en un par de columnas nuevas (por ejemplo: "ReviewSentiment", "TitleSentiment").

Paso 6. Prepare un dataframe con las columnas originales + las columnas creadas previamente, tendrían que haber 8 columnas, 3 de ellas deben ser numéricas (incluyendo "Rating").

# 3. ETAPA 3: MACHINE LEARNING.
Paso 1. Asigne a la variable X las columnas numéricas menos "Rating"; asigne a la variable Y la columna "Rating", seleccione únicamente las filas sin datos nulos (no elimine ni trate las filas con datos nulos, esas se usarán para predecir)

Paso 2. Divida en una muestra de entrenamiento y en una muestra de pruebas, estratifique en base a la proporción de la variable objetivo. El tamaño de la muestra para entrenamiento debe ser del 85%. Asigne una semilla para poder reproducir los resultados.

Paso 3. Entrene los siguientes modelos:
- KNN para clasificación
- SVM para clasificación
- RandomForest para clasificación

Paso 4. Evalúe el rendimiento de los modelos (puede usar accuracy) creados en el paso previo, muestre las predicciones realizadas y compare con las etiquetas reales.

Precisión del modelo KNN: 0.1984732824427481
Precisión del modelo SVM: 0.4961832061068702
Precisión del modelo RandomForest: 0.4961832061068702
En este caso, tanto el modelo SVM como el modelo RandomForest muestran una precisión similar, aproximadamente del 49.62%, mientras que el modelo KNN presenta una precisión más baja, alrededor del 19.85%.

En los tres modelos la precisión es relativamente baja. Si prestamos atención en las listas de predicciones vs el valor real, podemos darnos cuenta del mal comportamiento que tienen. 

Existen calificaciones cuyo valor real es superbajo; sin embargo, los tres modelos lo clasificas en las clases de 9 o 10.  Por otro lado, aceitan positivamente cuando la calificación es alta. 

Paso 5. Debido a que este es un problema de clasificación, pero hay varias clases que son originalmente numéricas, se puede aplicar también una métrica de evaluación para regresión. Aplique el RMSE a las predicciones y las etiquetas reales, analice el resultado


RMSE of KNeighborsClassifier: 2.29003816762086
RMSE of SVC: 2.6787259133931505
RMSE of RandomForestClassifier: 2.6787259133931505
En términos de RMSE, el modelo KNN tiene un rendimiento ligeramente mejor que los modelos SVM y RandomForest. Independientemente de los valores de estas métricas, considero que los modelos no tienen un desempeño aceptable.

Paso 6. Utilice el modelo que se comportó mejor para predecir el "Rating" de las filas que tienen ese dato nulo, revise manualmente si la calificación predicha es consistente con el comentario en la reseña.¶

Paso 7. Escriba sus conclusiones al respecto.
Podemos observar que el modelo esta clasificando positivamnete las reseñas, sin embargo, no es consistente, si bien la mayoria de las reseñas son positivas, hay un para como la del registro 500 y 450 cuyas reseñas expresan comentarios negativos, exiten otras que expresan comentarios neutros o con alguna queja, por lo que tal vez el valor 10 no seria la mejor clasificacion.

Debemos de tomar en cuenta que la clase SentimentIntensityAnalyzer se basa en un modelo preentrenado que asigna puntajes de intensidad de sentimiento a las palabras y calcula el sentimiento general del texto basándose en estos puntajes. Debido a la capacidad como seres humanos de expresar sentimientos de formas tan variados, podemos estar obteniendo puntajes desproporcionados o sesgados y quizá esto sea lo que provoque que el modelo se inclinen a predecir siempre una crítica como buena, cuando no lo es.
Es claro que el modelo se equivoca al clasificar una reseña en un puntaje alto, y no al revés, clasificando bajo.