"""
Módulo de Preprocesamiento y Limpieza de Datos para NYC Yellow Taxi Trip Records.

Este módulo implementa las reglas de negocio y correcciones de calidad de datos derivadas
del análisis exploratorio inicial (EDA). Centraliza la lógica para transformar datos crudos
en un conjunto limpio, consistente y listo para la ingeniería de características.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
from pathlib import Path
import sys
import os

# Configuración del entorno de ejecución y logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging import get_full_logger
from config.paths import LOGGER_NAME, RAW_DATA, PROCESSED_DATA
from config.settings import PREPROCESSING_PARAMS, LOG_LEVEL

warnings.filterwarnings('ignore')
logger = get_full_logger(name=LOGGER_NAME, log_level=LOG_LEVEL)

class TaxiDataPreprocessor:
    """
    Clase responsable de la higienización y validación del dataset de taxis.
    
    Implementa un pipeline secuencial de limpieza que aborda:
    1. Integridad estructural (columnas, tipos de datos).
    2. Coherencia temporal (fechas válidas, duraciones lógicas).
    3. Normalización categórica (mapeo de IDs, eliminación de valores no estándar).
    4. Consistencia financiera (tarifas no negativas, validación de sumas).
    5. Detección y tratamiento de outliers estadísticos.
    
    Esta clase no genera nuevas características (Feature Engineering), sino que asegura
    la calidad de los datos base sobre los cuales se construirán dichas características.
    """
    
    def __init__(self, raw_data_path: str = None):
        """
        Inicializa el preprocesador de datos.

        Args:
            raw_data_path (str, optional): Ruta al archivo de datos crudos.
                                           Por defecto utiliza la ruta definida en configuración.
        """
        self.raw_data_path = raw_data_path or RAW_DATA
        self.df = None
        self.original_shape = None
        self.preprocessing_stats = {}
        
        # Mapeos estándar según el diccionario de datos de la TLC (Taxi & Limousine Commission)
        self.vendor_mapping = {
            1: "Creative Mobile Technologies LLC",
            2: "Curb Mobility LLC", 
            6: "Myle Technologies Inc",
            7: "Helix"
        }
        
        self.ratecode_mapping = {
            1: "Standard rate", 2: "JFK", 3: "Newark",
            4: "Nassau or Westchester", 5: "Negotiated fare", 
            6: "Group ride", 99: "Null/unknown"
        }
        
        self.payment_mapping = {
            0: "Flex Fare trip", 1: "Credit card", 2: "Cash", 
            3: "No charge", 4: "Dispute", 5: "Unknown", 6: "Voided trip"
        }
        
        # Definición de campos que componen la estructura de costos del viaje
        self.monetary_fields = [
            'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
            'tolls_amount', 'improvement_surcharge', 'total_amount',
            'congestion_surcharge', 'airport_fee'
        ]
        
        logger.info("TaxiDataPreprocessor initialized")
    
    def load_data(self) -> pd.DataFrame:
        """
        Carga el dataset crudo desde el almacenamiento persistente.
        Registra las dimensiones iniciales para el seguimiento de la reducción de datos.
        """
        try:
            logger.info(f"Loading data from: {self.raw_data_path}")
            self.df = pd.read_parquet(self.raw_data_path)
            self.original_shape = self.df.shape
            logger.info(f"Data loaded: {self.original_shape[0]:,} rows, {self.original_shape[1]} columns")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_basic_structure(self) -> dict:
        """
        Verifica la integridad estructural del dataset contra el esquema esperado.
        Identifica columnas faltantes, tipos de datos incorrectos y evalúa la completitud (valores nulos).
        
        Returns:
            dict: Resumen estadístico del estado inicial de los datos.
        """
        logger.info("=== BASIC STRUCTURE VALIDATION ===")
        
        # Definición del esquema esperado según especificaciones del proyecto
        expected_columns = [
            'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
            'PULocationID', 'DOLocationID', 'payment_type',
            'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
            'improvement_surcharge', 'total_amount', 'congestion_surcharge',
            'airport_fee'
        ]
        
        actual_columns = set(self.df.columns)
        expected_set = set(expected_columns)
        
        missing_cols = expected_set - actual_columns
        extra_cols = actual_columns - expected_set
        
        validation_stats = {
            'total_records': len(self.df),
            'total_columns': len(actual_columns),
            'missing_columns': list(missing_cols),
            'extra_columns': list(extra_cols),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Análisis de completitud (Valores nulos)
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        validation_stats['missing_values'] = {
            col: {'count': int(count), 'percentage': round(pct, 2)}
            for col, count, pct in zip(missing_values.index, missing_values.values, missing_percentage.values)
            if count > 0
        }
        
        # Detección de duplicidad
        duplicates = self.df.duplicated().sum()
        validation_stats['duplicates'] = {
            'count': int(duplicates),
            'percentage': round(duplicates/len(self.df)*100, 2)
        }
        
        logger.info(f"Columnas faltantes: {missing_cols}")
        logger.info(f"Columnas adicionales: {extra_cols}")
        logger.info(f"Registros con valores faltantes: {sum(v['count'] for v in validation_stats['missing_values'].values())}")
        logger.info(f"Registros duplicados: {duplicates}")
        
        self.preprocessing_stats['validation'] = validation_stats
        return validation_stats
    
    def clean_datetime_fields(self) -> pd.DataFrame:
        """
        Aplica reglas de negocio para validar la coherencia temporal de los viajes.
        
        Acciones:
        1. Convierte columnas a objetos datetime.
        2. Filtra registros fuera del periodo de análisis (Mayo 2022).
        3. Calcula la duración del viaje y elimina registros con duraciones imposibles
           (negativas, cero, o excesivamente largas > 3 horas).
        """
        logger.info("=== DATETIME FIELDS CLEANING ===")
        
        initial_count = len(self.df)
        
        # Normalización de tipos de datos temporales
        datetime_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
        for col in datetime_cols:
            if col not in self.df.columns:
                continue
            
            if not self.df[col].dtype == 'object':
                continue
            
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Filtrado por ventana temporal del proyecto
        if 'tpep_pickup_datetime' in self.df.columns:
            valid_date_mask = (
                (self.df['tpep_pickup_datetime'].dt.year == PREPROCESSING_PARAMS['data_year']) &
                (self.df['tpep_pickup_datetime'].dt.month == PREPROCESSING_PARAMS['data_month'])
            )
            
            invalid_dates = (~valid_date_mask).sum()
            logger.info(f"Records with dates outside {PREPROCESSING_PARAMS['data_year']}-{PREPROCESSING_PARAMS['data_month']}: {invalid_dates:,} ({invalid_dates/len(self.df)*100:.2f}%)")
            
            if invalid_dates > 0:
                self.df = self.df[valid_date_mask].copy()
                logger.info(f"Filtered {invalid_dates:,} records with invalid dates")
        
        # Validación de duración lógica del viaje
        if all(col in self.df.columns for col in datetime_cols):
            self.df['trip_duration_minutes'] = (
                self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']
            ).dt.total_seconds() / 60
            
            # Filtro de plausibilidad: Duración entre 1 minuto y 3 horas
            duration_mask = (
                (self.df['trip_duration_minutes'] >= 1) &
                (self.df['trip_duration_minutes'] <= 180)
            )
            
            invalid_duration = (~duration_mask).sum()
            logger.info(f"Records with invalid duration: {invalid_duration:,} ({invalid_duration/len(self.df)*100:.2f}%)")
            
            if invalid_duration > 0:
                self.df = self.df[duration_mask].copy()
                logger.info(f"Filtered {invalid_duration:,} records with invalid duration")
            
        final_count = len(self.df)
        removed_records = initial_count - final_count
        
        logger.info(f"Removed records due to date/duration: {removed_records:,}")
        logger.info(f"Remaining records: {final_count:,}")
        
        self.preprocessing_stats['datetime_cleaning'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_records': removed_records,
            'removal_percentage': round(removed_records/initial_count*100, 2)
        }
        
        return self.df
    
    def clean_categorical_fields(self) -> pd.DataFrame:
        """
        Normaliza y limpia variables categóricas basándose en dominios de valores válidos.
        
        Acciones:
        1. Valida VendorID contra proveedores conocidos.
        2. Elimina registros con recuentos de pasajeros inválidos (0 o >6).
        3. Valida códigos de tarifa (RatecodeID) y métodos de pago.
        4. Normaliza banderas de operación (Store and Forward flag).
        """
        logger.info("=== CLEANING CATEGORICAL FIELDS ===")
        
        initial_count = len(self.df)
        
        # Validación de Proveedores
        if 'VendorID' in self.df.columns:
            valid_vendors = list(self.vendor_mapping.keys())
            invalid_vendor_mask = ~self.df['VendorID'].isin(valid_vendors)
            invalid_vendors = invalid_vendor_mask.sum()
            
            if invalid_vendors > 0:
                logger.info(f"Invalid VendorIDs: {invalid_vendors:,}")
                self.df = self.df[~invalid_vendor_mask].copy()
        
        # Validación de Pasajeros (Regla de negocio: Viaje debe tener pasajeros)
        if 'passenger_count' in self.df.columns:
            passenger_mask = (
                (self.df['passenger_count'] >= 1) &
                (self.df['passenger_count'] <= 6) &
                (self.df['passenger_count'].notna())
            )
            
            invalid_passengers = (~passenger_mask).sum()
            logger.info(f"Records with invalid passenger_count: {invalid_passengers:,}")
            
            if invalid_passengers > 0:
                self.df = self.df[passenger_mask].copy()
        
        # Validación de Códigos de Tarifa
        if 'RatecodeID' in self.df.columns:
            valid_ratecodes = list(self.ratecode_mapping.keys())
            invalid_ratecode_mask = (
                ~self.df['RatecodeID'].isin(valid_ratecodes) |
                self.df['RatecodeID'].isna()
            )
            invalid_ratecodes = invalid_ratecode_mask.sum()
            
            if invalid_ratecodes > 0:
                logger.info(f"Invalid RatecodeIDs: {invalid_ratecodes:,}")
                self.df = self.df[~invalid_ratecode_mask].copy()
        
        # Validación de Tipos de Pago
        if 'payment_type' in self.df.columns:
            valid_payments = list(self.payment_mapping.keys())
            invalid_payment_mask = ~self.df['payment_type'].isin(valid_payments)
            invalid_payments = invalid_payment_mask.sum()
            
            if invalid_payments > 0:
                logger.info(f"Invalid payment_types: {invalid_payments:,}")
                self.df = self.df[~invalid_payment_mask].copy()
        
        # Validación de Bandera de Almacenamiento
        if 'store_and_fwd_flag' in self.df.columns:
            valid_flags = ['Y', 'N']
            invalid_flag_mask = ~self.df['store_and_fwd_flag'].isin(valid_flags)
            invalid_flags = invalid_flag_mask.sum()
            
            if invalid_flags > 0:
                logger.info(f"Invalid store_and_fwd_flags: {invalid_flags:,}")
                self.df = self.df[~invalid_flag_mask].copy()
        
        final_count = len(self.df)
        removed_records = initial_count - final_count
        
        logger.info(f"Records removed due to categorical fields: {removed_records:,}")
        
        self.preprocessing_stats['categorical_cleaning'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_records': removed_records,
            'removal_percentage': round(removed_records/initial_count*100, 2)
        }
        
        return self.df
    
    def clean_monetary_fields(self) -> pd.DataFrame:
        """
        Asegura la consistencia financiera de los registros.
        
        Acciones:
        1. Elimina transacciones con valores monetarios negativos (posibles reversiones o errores).
        2. Descarta viajes con tarifa cero o distancia cero (datos no válidos para modelado).
        3. Corrige anomalías específicas detectadas en el EDA (ej. propinas en pagos en efectivo).
        4. Aplica truncamiento de valores extremos (Outliers) en la tarifa base.
        """
        logger.info("=== CLEANING MONETARY FIELDS ===")
        
        initial_count = len(self.df)
        
        # Validación de positividad en campos financieros y distancia
        positive_fields = ['fare_amount', 'trip_distance', 'total_amount', 'tip_amount', 'tolls_amount']
        
        for field in positive_fields:
            if not field in self.df.columns:
                continue
            negative_mask = self.df[field] < 0
            negative_count = negative_mask.sum()
            
            if negative_count == 0:
                continue
            
            logger.info(f"Negative values in {field}: {negative_count:,}")
            self.df = self.df[~negative_mask].copy()
        
        # Eliminación de registros sin costo (fare_amount = 0)
        if 'fare_amount' in self.df.columns:
            zero_fare_mask = self.df['fare_amount'] == 0
            zero_fares = zero_fare_mask.sum()
            
            if zero_fares > 0:
                logger.info(f"Trips with fare_amount = 0: {zero_fares:,}")
                self.df = self.df[~zero_fare_mask].copy()
        
        # Eliminación de registros sin desplazamiento (trip_distance = 0)
        if 'trip_distance' in self.df.columns:
            zero_distance_mask = self.df['trip_distance'] == 0
            zero_distance = zero_distance_mask.sum()
            
            if zero_distance > 0:
                logger.info(f"Trips with distance 0: {zero_distance:,}")
                self.df = self.df[~zero_distance_mask].copy()
        
        # Corrección de anomalía: Propinas registradas en pagos en efectivo
        # Por definición, el sistema no debería registrar propinas en efectivo.
        if 'tip_amount' in self.df.columns and 'payment_type' in self.df.columns:
            cash_tip_mask = (self.df['payment_type'] == 2) & (self.df['tip_amount'] > 0)
            cash_tips = cash_tip_mask.sum()
            
            if cash_tips > 0:
                logger.info(f"Anomalous tips in cash payments: {cash_tips:,}")
                self.df.loc[cash_tip_mask, 'tip_amount'] = 0
        
        # Filtrado de outliers extremos en tarifas (Protección estadística)
        if 'fare_amount' in self.df.columns:
            fare_q95 = self.df['fare_amount'].quantile(0.95)
            # Umbral conservador: 3 veces el percentil 95
            high_fare_mask = self.df['fare_amount'] > fare_q95 * 3
            high_fares = high_fare_mask.sum()
            
            if high_fares > 0:
                logger.info(f"Extremely high fares removed: {high_fares:,}")
                self.df = self.df[~high_fare_mask].copy()
        
        final_count = len(self.df)
        removed_records = initial_count - final_count
        
        logger.info(f"Records removed due to monetary fields: {removed_records:,}")
        
        self.preprocessing_stats['monetary_cleaning'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_records': removed_records,
            'removal_percentage': round(removed_records/initial_count*100, 2)
        }
        
        return self.df
    
    def validate_total_amount(self) -> pd.DataFrame:
        """
        Verifica la consistencia aritmética del campo 'total_amount'.
        
        Compara el valor reportado en 'total_amount' con la suma calculada de sus componentes
        (tarifa, impuestos, recargos, propinas, peajes). Si existe discrepancia, se corrige
        el valor total para garantizar integridad contable.
        """
        logger.info("=== VALIDATION OF total_amount ===")
        
        if 'total_amount' not in self.df.columns:
            logger.warning("total_amount field not found")
            return self.df
        
        # Definición de componentes de la suma
        base_fields = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge']
        additional_fields = ['congestion_surcharge', 'airport_fee']
        
        # Cálculo de la suma esperada
        calculated_total = pd.Series(0, index=self.df.index)
        
        for field in base_fields:
            if field not in self.df.columns:
                continue
            calculated_total += self.df[field].fillna(0)
        
        for field in additional_fields:
            if field not in self.df.columns:
                continue
            calculated_total += self.df[field].fillna(0)
        
        # Detección de discrepancias (tolerancia de 1 centavo por errores de punto flotante)
        diff = abs(self.df['total_amount'] - calculated_total)
        discrepancy_mask = diff > 0.01 
        discrepancies = discrepancy_mask.sum()
        
        logger.info(f"Records with discrepancies in total_amount: {discrepancies:,} ({discrepancies/len(self.df)*100:.2f}%)")
        
        if discrepancies > 0:
            logger.info(f"Average difference: ${diff[discrepancy_mask].mean():.2f}")
            logger.info(f"Maximum difference: ${diff.max():.2f}")
            
            # Corrección automática del total
            self.df.loc[discrepancy_mask, 'total_amount'] = calculated_total[discrepancy_mask]
            logger.info("total_amount corrected based on sum of components")
        
        self.preprocessing_stats['total_amount_validation'] = {
            'discrepancies_found': int(discrepancies),
            'discrepancy_percentage': round(discrepancies/len(self.df)*100, 2),
            'avg_difference': round(diff[discrepancy_mask].mean(), 2) if discrepancies > 0 else 0,
            'max_difference': round(diff.max(), 2)
        }
        
        return self.df
    
    def remove_outliers(self, method='iqr', factor=1.5) -> pd.DataFrame:
        """
        Aplica técnicas estadísticas para la detección y eliminación de outliers en variables numéricas clave.
        Utiliza el método del Rango Intercuartílico (IQR) para identificar valores anómalos.
        
        Args:
            method (str): Metodología de detección ('iqr').
            factor (float): Multiplicador del IQR para definir los límites (default 1.5 para outliers moderados).
        """
        logger.info(f"=== OUTLIER REMOVAL ({method.upper()}) ===")
        
        initial_count = len(self.df)
        
        # Variables sujetas a limpieza de outliers
        outlier_fields = ['trip_distance', 'fare_amount', 'total_amount', 'tip_amount', 'trip_duration_minutes']
        outlier_fields = [field for field in outlier_fields if field in self.df.columns]
        
        total_outliers_removed = 0
        
        for field in outlier_fields:
            if method == 'iqr':
                Q1 = self.df[field].quantile(0.25)
                Q3 = self.df[field].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outlier_mask = (self.df[field] < lower_bound) | (self.df[field] > upper_bound)
                outliers_count = outlier_mask.sum()
                
                if outliers_count > 0:
                    logger.info(f"Outliers in {field}: {outliers_count:,} ({outliers_count/len(self.df)*100:.2f}%)")
                    self.df = self.df[~outlier_mask].copy()
                    total_outliers_removed += outliers_count
        
        final_count = len(self.df)
        
        logger.info(f"Total outliers removed: {total_outliers_removed:,}")
        logger.info(f"Remaining records: {final_count:,}")
        
        self.preprocessing_stats['outlier_removal'] = {
            'method': method,
            'factor': factor,
            'initial_count': initial_count,
            'final_count': final_count,
            'outliers_removed': total_outliers_removed,
            'removal_percentage': round(total_outliers_removed/initial_count*100, 2)
        }
        
        return self.df
    
    def get_preprocessing_summary(self) -> dict:
        """
        Consolida y retorna las estadísticas acumuladas de todo el proceso de limpieza.
        Útil para auditoría de calidad de datos y logging.
        """
        if not self.preprocessing_stats:
            logger.warning("No preprocessing statistics available")
            return {}
        
        summary = {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape if self.df is not None else None,
            'total_removed': self.original_shape[0] - (self.df.shape[0] if self.df is not None else 0),
            'removal_percentage': round((self.original_shape[0] - (self.df.shape[0] if self.df is not None else 0)) / self.original_shape[0] * 100, 2),
            'steps': self.preprocessing_stats
        }
        
        return summary
    
    def preprocess_full_pipeline(self, 
                                 remove_outliers: bool = True,
                                 outlier_method: str = 'iqr',
                                 outlier_factor: float = 1.5) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de preprocesamiento de manera secuencial.
        
        Args:
            remove_outliers (bool): Bandera para activar la eliminación estadística de outliers.
            outlier_method (str): Método estadístico a utilizar.
            outlier_factor (float): Sensibilidad del método de outliers.
            
        Returns:
            pd.DataFrame: Dataset limpio y validado.
        """
        logger.info("=== STARTING DATA CLEANING PIPELINE ===")
        
        # 1. Ingesta de datos
        self.load_data()
        
        # 2. Validación estructural inicial
        self.validate_basic_structure()
        
        # 3. Limpieza de componente temporal
        self.clean_datetime_fields()
        
        # 4. Normalización de variables categóricas
        self.clean_categorical_fields()
        
        # 5. Saneamiento de variables monetarias
        self.clean_monetary_fields()
        
        # 6. Verificación de consistencia aritmética
        self.validate_total_amount()
        
        # 7. Eliminación de ruido estadístico (Outliers)
        if remove_outliers:
            self.remove_outliers(method=outlier_method, factor=outlier_factor)
        
        # Generación de reporte final
        summary = self.get_preprocessing_summary()
        logger.info("=== DATA CLEANING COMPLETED ===")
        logger.info(f"Original records: {summary['original_shape'][0]:,}")
        logger.info(f"Clean records: {summary['total_removed']:,} ({summary['removal_percentage']:.2f}%)")
        logger.info("Data ready for feature engineering")
        
        return self.df
    
    def save_processed_data(self, output_path: str = None) -> str:
        """
        Persiste el dataset limpio en formato Parquet para su posterior consumo.
        
        Args:
            output_path (str, optional): Ruta de destino personalizada.
            
        Returns:
            str: Ruta absoluta del archivo generado.
        """
        if self.df is None:
            raise ValueError("No processed data to save. Run the pipeline first.")
        else:
            logger.info("Saving processed data...")
        
        output_path = output_path or PROCESSED_DATA
        
        # Garantiza la existencia del directorio de destino
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Escritura optimizada
        self.df.to_parquet(output_path, index=False, engine='pyarrow', compression='gzip')
        logger.info(f"Processed data saved at: {output_path}")
        logger.info(f"File size: {Path(output_path).stat().st_size / 1024**2:.2f} MB")
        
        return output_path


def main():
    """
    Función principal de ejecución del script de limpieza.
    Instancia el preprocesador, ejecuta el pipeline y reporta resultados.
    """
    try:
        # Inicialización del componente
        preprocessor = TaxiDataPreprocessor()
        
        # Ejecución del flujo de trabajo con parámetros estándar
        cleaned_data = preprocessor.preprocess_full_pipeline(
            remove_outliers=True,
            outlier_method='iqr',
            outlier_factor=1.5
        )
        
        # Persistencia de resultados
        output_path = preprocessor.save_processed_data()
        
        # Reporte de métricas de calidad de datos
        summary = preprocessor.get_preprocessing_summary()
        print("\n" + "="*50)
        print("DATA CLEANING SUMMARY")
        print("="*50)
        print(f"Original records: {summary['original_shape'][0]:,}")
        print(f"Clean records: {summary['final_shape'][0]:,}")
        print(f"Records removed: {summary['total_removed']:,} ({summary['removal_percentage']:.2f}%)")
        print(f"Clean data file: {output_path}")
        print("\nNEXT STEP:")
        print("   Run features.py for feature engineering")
        
        return cleaned_data, summary
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        raise


if __name__ == "__main__":
    main()