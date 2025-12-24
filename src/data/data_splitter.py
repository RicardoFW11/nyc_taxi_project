import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

from src.data.feature_selector import FeatureSelector
from src.config.settings import RANDOM_STATE, TEST_SIZE, VAL_SIZE, TRAIN_SIZE
from src.config.paths import LOGGER_NAME

from src.utils.logging import LoggerFactory

class DataSplitter:
    def __init__(self, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE):
        """
        Orquesta la partición del conjunto de datos para el entrenamiento, validación y prueba
        de los modelos de estimación de tarifa y duración.
        
        Garantiza que ambos modelos utilicen exactamente los mismos índices de filas para
        cada subconjunto, asegurando una comparabilidad justa en las métricas de evaluación.

        Parámetros:
        -----------
        test_size : float
            Proporción del dataset total que se reservará exclusivamente para la evaluación final (Hold-out set).
        val_size : float
            Proporción del dataset total destinada al ajuste de hiperparámetros y parada temprana.
        random_state : int
            Semilla aleatoria para garantizar la reproducibilidad determinista de los cortes.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        self.logging = LoggerFactory.create_logger(
            name=LOGGER_NAME,
            log_level='DEBUG',
            console_output=True,
            file_output=False
        )
        
        # Verifica la coherencia matemática de las proporciones de partición.
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Ejecuta verificaciones de integridad sobre el DataFrame antes de procesarlo.
        Asegura que existan las columnas objetivo y que el volumen de datos sea suficiente.
        """
        required_columns = ['fare_amount', 'trip_duration_minutes']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Analiza la calidad de las variables objetivo para advertir sobre
        # una posible pérdida excesiva de información si hay demasiados nulos.
        for col in required_columns:
            missing_pct = df[col].isna().mean()
            if missing_pct > 0.5:
                self.logging.warning(f"High missing values in {col}: {missing_pct:.2%}")
                
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera la matriz de características (Features) eliminando las variables objetivo
        y aquellas columnas que podrían introducir 'Data Leakage' (fugas de información).
        """
        exclude_columns = ['fare_amount', 'trip_duration_minutes']
        
        # Excluye explícitamente variables que contienen información del futuro 
        # (como la fecha real de llegada) o componentes directos del precio final
        # (propinas, peajes, monto total) para mantener la honestidad del modelo.
        additional_exclude = [
            'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'total_amount', 'tip_amount', 'tolls_amount'
        ]
        
        exclude_columns.extend([col for col in additional_exclude if col in df.columns])
        X_all = df.drop(columns=exclude_columns)
        
        return X_all
                
    def split_data_for_both_models(self, df: pd.DataFrame, 
                                 sample_for_feature_selection: int = 10000) -> Dict:
        """
        Ejecuta el flujo completo de preparación: limpieza, división estratificada y
        selección de características específica para cada objetivo.

        Estrategia:
        1. Limpia registros sin target válido.
        2. Divide Train/Val/Test manteniendo los mismos índices para ambos problemas.
        3. Realiza una selección de características independiente para 'fare' y 'duration',
           ya que los predictores relevantes pueden variar entre costo y tiempo.
        
        Parámetros:
        -----------
        df : pd.DataFrame
            Conjunto de datos completo con características y objetivos.
        sample_for_feature_selection : int
            Número máximo de muestras a utilizar durante la etapa de selección de características
            para optimizar el tiempo de cómputo sin sacrificar precisión estadística.
        
        Retorna:
        --------
        Dict
            Diccionario estructurado conteniendo los datasets divididos (X_train, y_train, etc.)
            y metadatos sobre el proceso para cada modelo.
        """
        
        # Ejecuta validaciones de estructura e integridad.
        self._validate_data(df)
        
        # Descarta registros que carecen de valores en las variables objetivo,
        # ya que no aportan al aprendizaje supervisado.
        df_clean = df.dropna(subset=['fare_amount', 'trip_duration_minutes'])
        
        if len(df_clean) < len(df):
            self.logging.info(f"Removed {len(df) - len(df_clean)} rows with missing targets")
        
        # Separa la matriz de características de los vectores objetivo.
        X_all = self._prepare_features(df_clean)
        y_fare = df_clean['fare_amount']
        y_duration = df_clean['trip_duration_minutes']
        
        logging.info(f"Total samples: {len(df_clean)}")
        logging.info(f"Total features available: {len(X_all.columns)}")
        
        # Fase 1: Segregación del conjunto de prueba (Test set).
        # Se aísla primero para garantizar que nunca sea visto durante el entrenamiento.
        X_temp, X_test, y_fare_temp, y_fare_test, y_duration_temp, y_duration_test = train_test_split(
            X_all, y_fare, y_duration,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Fase 2: División del conjunto restante en Entrenamiento y Validación.
        # Se ajusta el tamaño relativo de validación respecto al remanente.
        val_size_adjusted = self.val_size / (1 - self.test_size)
        
        X_train, X_val, y_fare_train, y_fare_val, y_duration_train, y_duration_val = train_test_split(
            X_temp, y_fare_temp, y_duration_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state
        )
        
        self.logging.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Optimización: Utiliza una muestra representativa para el cálculo de importancia de features.
        # Esto reduce drásticamente el tiempo de ejecución en algoritmos como Boruta o RFE.
        sample_size = min(sample_for_feature_selection, len(X_train))
        if sample_size < len(X_train):
            sample_idx = np.random.RandomState(self.random_state).choice(
                len(X_train), size=sample_size, replace=False
            )
            X_sample = X_train.iloc[sample_idx]
            y_fare_sample = y_fare_train.iloc[sample_idx]
            y_duration_sample = y_duration_train.iloc[sample_idx]
        else:
            X_sample = X_train
            y_fare_sample = y_fare_train
            y_duration_sample = y_duration_train
        
        # Inicializa los selectores especializados para cada variable dependiente.
        fare_selector = FeatureSelector('fare_amount')
        duration_selector = FeatureSelector('trip_duration_minutes')
        
        # Ejecuta la identificación de las variables más predictivas para la tarifa.
        self.logging.info("Selecting features for fare model...")
        fare_features, fare_scores = fare_selector.select_features_for_fare(
            X_sample, y_fare_sample
        )
        
        # Ejecuta la identificación de las variables más predictivas para la duración.
        self.logging.info("Selecting features for duration model...")
        duration_features, duration_scores = duration_selector.select_features_for_duration(
            X_sample, y_duration_sample
        )
        
        self.logging.info(f"Selected {len(fare_features)} features for fare model")
        self.logging.info(f"Selected {len(duration_features)} features for duration model")
        
        # Empaqueta los subconjuntos filtrados por las columnas seleccionadas para el modelo de tarifa.
        fare_data = {
            'X_train': X_train[fare_features],
            'X_val': X_val[fare_features],
            'X_test': X_test[fare_features],
            'y_train': y_fare_train,
            'y_val': y_fare_val,
            'y_test': y_fare_test,
            'features': fare_features,
            'feature_scores': fare_scores,
            'selector': fare_selector
        }
        
        # Empaqueta los subconjuntos filtrados por las columnas seleccionadas para el modelo de duración.
        duration_data = {
            'X_train': X_train[duration_features],
            'X_val': X_val[duration_features],
            'X_test': X_test[duration_features],
            'y_train': y_duration_train,
            'y_val': y_duration_val,
            'y_test': y_duration_test,
            'features': duration_features,
            'feature_scores': duration_scores,
            'selector': duration_selector
        }
        
        return {
            'fare_model': fare_data,
            'duration_model': duration_data,
            'split_info': {
                'total_samples': len(df_clean),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'train_ratio': len(X_train) / len(df_clean),
                'val_ratio': len(X_val) / len(df_clean),
                'test_ratio': len(X_test) / len(df_clean),
                'feature_selection_sample_size': sample_size
            }
        }
        
    def get_feature_overlap(self, splits_dict: Dict) -> Dict:
        """
        Analiza la intersección y divergencia de características entre ambos modelos.
        Útil para diagnóstico y para entender qué variables son universalmente predictivas
        versus cuáles son específicas para un solo objetivo.
        """
        fare_features = set(splits_dict['fare_model']['features'])
        duration_features = set(splits_dict['duration_model']['features'])
        
        overlap = fare_features.intersection(duration_features)
        fare_only = fare_features - duration_features
        duration_only = duration_features - fare_features
        
        return {
            'overlap_features': list(overlap),
            'fare_only_features': list(fare_only),
            'duration_only_features': list(duration_only),
            'overlap_count': len(overlap),
            'fare_only_count': len(fare_only),
            'duration_only_count': len(duration_only)
        }