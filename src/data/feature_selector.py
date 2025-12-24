import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:
    def __init__(self, target_name: str):
        """
        Clase responsable de identificar y seleccionar el subconjunto óptimo de variables predictoras.
        Implementa una estrategia híbrida que combina métodos de filtrado estadístico univariante
        con la importancia de características derivada de modelos basados en árboles (Random Forest).

        Parámetros:
        -----------
        target_name : str
            Nombre de la variable objetivo (target) para la cual se optimizará la selección.
        """
        self.target_name = target_name
        self.selected_features = None
        
    def select_features_for_fare(self, X: pd.DataFrame, y: pd.Series, k=30):
        """
        Ejecuta el proceso de selección de características optimizado para la predicción de tarifas (Fare Amount).
        
        Aplica un filtro estricto sobre las variables candidatas para eliminar cualquier característica
        que constituya 'Data Leakage' (información disponible solo post-viaje). Posteriormente,
        ranking y selecciona las 'k' mejores variables basándose en un puntaje combinado.

        Parámetros:
        -----------
        X : pd.DataFrame
            Matriz de características candidatas.
        y : pd.Series
            Vector de la variable objetivo (monto de la tarifa).
        k : int
            Número máximo de características a conservar en el modelo final.
            
        Retorna:
        --------
        tuple
            - Lista de nombres de las características seleccionadas.
            - DataFrame con el detalle de los puntajes de importancia para cada variable evaluada.
        """
        
        # Definición explícita de variables candidatas.
        # Se han auditado y excluido variables que contienen información futura (Leakage)
        # o componentes directos de la tarifa final (peajes, recargos, impuestos).
        fare_relevant_features = [
            # --- CARACTERÍSTICAS VÁLIDAS (Información ex-ante) ---

            # Métricas básicas del viaje
            'trip_distance',        # Distancia estimada de la ruta óptima.
            'log_trip_distance',    # Transformación logarítmica para manejar la asimetría.
            'passenger_count',      # Declarado por el usuario al inicio.

            # Características Temporales (Derivadas del Timestamp de inicio)
            'pickup_hour', 
            'pickup_day_of_week', 
            'is_weekend', 
            'is_rush_hour', 
            'is_morning_rush', 
            'is_evening_rush',
            'time_of_day_Morning', 
            'time_of_day_Afternoon', 
            'time_of_day_Evening',

            # Características Geoespaciales (Origen y Destino)
            'is_airport_trip', 
            'is_jfk_trip', 
            'is_newark_trip',
            'pickup_popularity',    # Frecuencia histórica de la zona.
            'dropoff_popularity',

            # Información de Tarifas y Pagos (Conocida al configurar el viaje)
            'is_standard_rate', 
            'ratecode_name_Standard', 
            'ratecode_name_Newark',
            'is_credit_card', 
            'is_cash_payment', 
            'payment_name_Credit_Card',

            # Categorización del Viaje (Estimaciones pre-viaje)
            'distance_category_encoded', 
            'is_short_distance', 
            'is_medium_distance', 
            'is_long_distance',
            
            # Agregaciones Históricas (Contexto estadístico, no del viaje actual)
            'hourly_fare_amount_mean', 
            'hourly_fare_amount_std',
            'daily_avg_fare_amount'

            # --- VARIABLES EXCLUIDAS POR DATA LEAKAGE ---
            # 'fare_per_mile': Contiene la variable objetivo implícita.
            # 'trip_duration_minutes': Información conocida solo al finalizar.
            # 'avg_speed_mph': Requiere conocer la duración real.
            # 'extra', 'mta_tax', 'improvement_surcharge', etc.: Componentes de facturación post-viaje.
        ]

        # Intersección entre características teóricas y disponibles en el dataset
        available_features = [f for f in fare_relevant_features if f in X.columns]
        
        # Método 1: Selección Estadística Univariante (F-test para regresión)
        # Captura relaciones lineales directas entre cada feature y el target.
        selector_stats = SelectKBest(score_func=f_regression, k=min(k, len(available_features)))
        _ = selector_stats.fit_transform(X[available_features], y)
        
        # Método 2: Importancia basada en Modelos (Random Forest)
        # Captura relaciones no lineales e interacciones complejas entre variables.
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[available_features], y)
        
        # Consolidación de resultados
        feature_scores = pd.DataFrame({
            'feature': available_features,
            'rf_importance': rf.feature_importances_,
            'statistical_score': selector_stats.scores_
        })
        
        # Normalización Min-Max y cálculo de puntaje de consenso
        feature_scores['rf_norm'] = feature_scores['rf_importance'] / feature_scores['rf_importance'].max()
        feature_scores['stat_norm'] = feature_scores['statistical_score'] / feature_scores['statistical_score'].max()
        feature_scores['combined_score'] = (feature_scores['rf_norm'] + feature_scores['stat_norm']) / 2
        
        # Selección final basada en el puntaje combinado
        top_features = feature_scores.nlargest(k, 'combined_score')['feature'].tolist()
        
        self.selected_features = top_features
        return top_features, feature_scores
    
    def select_features_for_duration(self, X: pd.DataFrame, y: pd.Series, k=25):
        """
        Ejecuta el proceso de selección de características optimizado para la predicción de la duración (Trip Duration).
        
        La lógica difiere del modelo de tarifas, priorizando variables que afectan el flujo de tráfico
        y la velocidad operativa, manteniendo estrictas exclusiones para evitar fugas de información.

        Parámetros:
        -----------
        X : pd.DataFrame
            Matriz de características candidatas.
        y : pd.Series
            Vector de la variable objetivo (duración en minutos).
        k : int
            Número máximo de características a conservar.
        """
        
        # Definición de variables candidatas auditadas para el modelo de tiempo.
        duration_relevant_features = [
            # Métricas básicas (Distancia física y volumen de pasajeros)
            'trip_distance', 
            'log_trip_distance',
            'passenger_count',
            
            # Variables Temporales (Críticas para modelar congestión)
            'pickup_hour', 'pickup_day_of_week', 'is_weekend',
            'is_rush_hour', 'is_morning_rush', 'is_evening_rush',
            'time_of_day_Morning', 'time_of_day_Afternoon', 'time_of_day_Evening',
            'detailed_time_category_Morning', 'detailed_time_category_Evening',
            'detailed_time_category_Late_Night', 'detailed_time_category_Night',
            
            # Variables Geoespaciales (Zonas de alta densidad o aeropuertos)
            'is_airport_trip', 'is_jfk_trip', 'is_newark_trip',
            'pickup_popularity', 'dropoff_popularity',
            
            # Categorización de Distancia
            'distance_category_encoded', 'is_short_distance',
            'is_medium_distance', 'is_long_distance',
            'is_very_short_trip', 'is_long_trip',
            
            # Información Operativa
            'VendorID', 'is_store_forward',
            'ratecode_name_Standard', 'ratecode_name_Newark',
            
            # Interacciones y Agregaciones Históricas
            'weekend_rush', 
            'airport_morning', 'airport_evening',
            'hourly_trip_distance_mean', 
            'daily_avg_avg_speed_mph', # Válido solo si representa el promedio histórico del día, no del viaje actual.
            'is_round_trip'

            # --- VARIABLES EXCLUIDAS POR DATA LEAKAGE ---
            # 'fare_amount': Altamente correlacionada pero desconocida al inicio exacto (si hay taxímetro).
            # 'avg_speed_mph': Derivada directa de distancia/tiempo real.
            # 'speed_category_encoded', 'is_slow_trip': Derivadas de velocidad real.
            # 'long_distance_long_time': Interacción que revela la duración.
        ]

        # Filtrado de características disponibles en el dataset actual
        available_features = [f for f in duration_relevant_features if f in X.columns]
        
        # Aplicación de la metodología híbrida (Estadística + Random Forest)
        selector_stats = SelectKBest(score_func=f_regression, k=min(k, len(available_features)))
        _ = selector_stats.fit_transform(X[available_features], y)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[available_features], y)
        
        # Consolidación y Normalización de Puntajes
        feature_scores = pd.DataFrame({
            'feature': available_features,
            'rf_importance': rf.feature_importances_,
            'statistical_score': selector_stats.scores_
        })
        
        feature_scores['rf_norm'] = feature_scores['rf_importance'] / feature_scores['rf_importance'].max()
        feature_scores['stat_norm'] = feature_scores['statistical_score'] / feature_scores['statistical_score'].max()
        feature_scores['combined_score'] = (feature_scores['rf_norm'] + feature_scores['stat_norm']) / 2
        
        # Selección del Top-K
        top_features = feature_scores.nlargest(k, 'combined_score')['feature'].tolist()
        
        self.selected_features = top_features
        return top_features, feature_scores

    def get_selected_features(self):
        """Devuelve la lista de características seleccionadas en la última ejecución."""
        return self.selected_features
    
    def transform(self, X: pd.DataFrame):
        """
        Reduce el DataFrame de entrada conservando únicamente las columnas seleccionadas previamente.
        
        Parámetros:
        -----------
        X : pd.DataFrame
            DataFrame original con todas las características.
            
        Retorna:
        --------
        pd.DataFrame
            DataFrame filtrado conteniendo solo las características relevantes.
            
        Raises:
        -------
        ValueError
            Si no se ha ejecutado un proceso de selección previo.
        """
        if self.selected_features is None:
            raise ValueError("No features selected. Run select_features first.")
        
        return X[self.selected_features]