"""
Módulo de preprocesamiento de datos para NYC Yellow Taxi Trip Records.
Basado en hallazgos del EDA realizado en 01_eda.ipynb.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
from pathlib import Path
import sys
import os

# Agregar src al path para imports relativos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging import get_full_logger
from config.paths import LOGGER_NAME, RAW_DATA, PROCESSED_DATA
from config.settings import PREPROCESSING_PARAMS

warnings.filterwarnings('ignore')
logger = get_full_logger(name=LOGGER_NAME, log_level="DEBUG")


class TaxiDataPreprocessor:
    """
    Clase para limpiar los datos de NYC Yellow Taxi basado en hallazgos del EDA.
    
    Responsabilidades:
    - Validar estructura básica del dataset
    - Limpiar campos temporales (fechas inválidas, duraciones anómalas)
    - Limpiar campos categóricos (valores inválidos, outliers)
    - Limpiar campos monetarios (valores negativos, discrepancias)
    - Validar y corregir total_amount
    - Remover outliers extremos
    
    Issues críticos identificados en EDA:
    - 28.01% registros con discrepancias en total_amount
    - Valores negativos generalizados (0.57% tarifas, 0.58% totales)
    - VendorID=5 inválido (14 registros)
    - Fechas fuera de rango mayo 2022
    - 73,587 viajes con 0 pasajeros (2.05%)
    - 46,438 viajes con distancia 0 (1.29%)
    - 135 propinas anómalas en pagos en efectivo
    
    Nota: La ingeniería de características se realiza en TaxiFeatureEngineer.
    """
    
    def __init__(self, raw_data_path: str = None):
        """
        Inicializar el preprocesador.
        
        Args:
            raw_data_path: Ruta al archivo de datos raw. Si None, usa RAW_DATA de config.
        """
        self.raw_data_path = raw_data_path or RAW_DATA
        self.df = None
        self.original_shape = None
        self.preprocessing_stats = {}
        
        # Definir mappings según diccionario de datos TLC
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
        
        # Definir campos monetarios según diccionario TLC
        self.monetary_fields = [
            'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
            'tolls_amount', 'improvement_surcharge', 'total_amount',
            'congestion_surcharge', 'airport_fee'
        ]
        
        logger.info("TaxiDataPreprocessor inicializado")
    
    def load_data(self) -> pd.DataFrame:
        """Cargar datos raw desde archivo parquet."""
        try:
            logger.info(f"Cargando datos desde: {self.raw_data_path}")
            self.df = pd.read_parquet(self.raw_data_path)
            self.original_shape = self.df.shape
            logger.info(f"Datos cargados: {self.original_shape[0]:,} filas, {self.original_shape[1]} columnas")
            return self.df
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise
    
    def validate_basic_structure(self) -> dict:
        """
        Validar estructura básica y completitud del dataset.
        
        Returns:
            dict: Estadísticas de validación
        """
        logger.info("=== VALIDACIÓN DE ESTRUCTURA BÁSICA ===")
        
        # Columnas esperadas según diccionario TLC
        expected_columns = [
            'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
            'PULocationID', 'DOLocationID', 'payment_type',
            'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
            'improvement_surcharge', 'total_amount', 'congestion_surcharge',
            'airport_fee'#, 'cbd_congestion_fee'
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
        
        # Valores faltantes
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        validation_stats['missing_values'] = {
            col: {'count': int(count), 'percentage': round(pct, 2)}
            for col, count, pct in zip(missing_values.index, missing_values.values, missing_percentage.values)
            if count > 0
        }
        
        # Duplicados
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
        Limpiar campos de fecha/hora basado en hallazgos del EDA.
        
        Issues identificados:
        - Fechas fuera del rango mayo 2022 (datos 2003-2022)
        - 3,030 viajes con duración negativa/cero (0.08%)
        - 48,944 viajes < 1 minuto (1.36%)
        - 5,084 viajes > 3 horas (0.14%)
        """
        logger.info("=== LIMPIEZA DE CAMPOS TEMPORALES ===")
        
        initial_count = len(self.df)
        
        # Convertir a datetime si es necesario
        datetime_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
        for col in datetime_cols:
            if col not in self.df.columns:
                continue
            
            if not self.df[col].dtype == 'object':
                continue
            
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Filtrar fechas válidas usando PREPROCESSING_PARAMS[data_year] y PREPROCESSING_PARAMS[data_month]
        if 'tpep_pickup_datetime' in self.df.columns:
            valid_date_mask = (
                (self.df['tpep_pickup_datetime'].dt.year == PREPROCESSING_PARAMS['data_year']) &
                (self.df['tpep_pickup_datetime'].dt.month == PREPROCESSING_PARAMS['data_month'])
            )
            
            invalid_dates = (~valid_date_mask).sum()
            logger.info(f"Registros con fechas fuera {PREPROCESSING_PARAMS['data_year']}-{PREPROCESSING_PARAMS['data_month']}: {invalid_dates:,} ({invalid_dates/len(self.df)*100:.2f}%)")
            
            if invalid_dates > 0:
                self.df = self.df[valid_date_mask].copy()
                logger.info(f"Filtrados {invalid_dates:,} registros con fechas inválidas")
        
        # Calcular duración del viaje
        if all(col in self.df.columns for col in datetime_cols):
            self.df['trip_duration_minutes'] = (
                self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']
            ).dt.total_seconds() / 60
            
            # Filtrar duraciones válidas (entre 1 minuto y 3 horas)
            duration_mask = (
                (self.df['trip_duration_minutes'] >= 1) &
                (self.df['trip_duration_minutes'] <= 180)
            )
            
            invalid_duration = (~duration_mask).sum()
            logger.info(f"Registros con duración inválida: {invalid_duration:,} ({invalid_duration/len(self.df)*100:.2f}%)")
            
            if invalid_duration > 0:
                self.df = self.df[duration_mask].copy()
                logger.info(f"Filtrados {invalid_duration:,} registros con duración inválida")
            
            # Solo crear duración del viaje (necesaria para validación)
            # Las features temporales se crean en features.py
        
        final_count = len(self.df)
        removed_records = initial_count - final_count
        
        logger.info(f"Registros removidos por fechas/duración: {removed_records:,}")
        logger.info(f"Registros restantes: {final_count:,}")
        
        self.preprocessing_stats['datetime_cleaning'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_records': removed_records,
            'removal_percentage': round(removed_records/initial_count*100, 2)
        }
        
        return self.df
    
    def clean_categorical_fields(self) -> pd.DataFrame:
        """
        Limpiar campos categóricos basado en diccionario TLC.
        
        Issues identificados:
        - VendorID=5 inválido (14 registros)
        - 73,587 viajes con 0 pasajeros (2.05%)
        - Valores faltantes en campos categóricos (3.6%)
        """
        logger.info("=== LIMPIEZA DE CAMPOS CATEGÓRICOS ===")
        
        initial_count = len(self.df)
        
        # Limpiar VendorID - mantener solo valores válidos
        if 'VendorID' in self.df.columns:
            valid_vendors = list(self.vendor_mapping.keys())
            invalid_vendor_mask = ~self.df['VendorID'].isin(valid_vendors)
            invalid_vendors = invalid_vendor_mask.sum()
            
            if invalid_vendors > 0:
                logger.info(f"VendorIDs inválidos: {invalid_vendors:,}")
                self.df = self.df[~invalid_vendor_mask].copy()
        
        # Limpiar passenger_count - remover 0 pasajeros y valores extremos
        if 'passenger_count' in self.df.columns:
            passenger_mask = (
                (self.df['passenger_count'] >= 1) &
                (self.df['passenger_count'] <= 6) &
                (self.df['passenger_count'].notna())
            )
            
            invalid_passengers = (~passenger_mask).sum()
            logger.info(f"Registros con passenger_count inválido: {invalid_passengers:,}")
            
            if invalid_passengers > 0:
                self.df = self.df[passenger_mask].copy()
        
        # Validar RatecodeID
        if 'RatecodeID' in self.df.columns:
            valid_ratecodes = list(self.ratecode_mapping.keys())
            invalid_ratecode_mask = (
                ~self.df['RatecodeID'].isin(valid_ratecodes) |
                self.df['RatecodeID'].isna()
            )
            invalid_ratecodes = invalid_ratecode_mask.sum()
            
            if invalid_ratecodes > 0:
                logger.info(f"RatecodeIDs inválidos: {invalid_ratecodes:,}")
                self.df = self.df[~invalid_ratecode_mask].copy()
        
        # Validar payment_type
        if 'payment_type' in self.df.columns:
            valid_payments = list(self.payment_mapping.keys())
            invalid_payment_mask = ~self.df['payment_type'].isin(valid_payments)
            invalid_payments = invalid_payment_mask.sum()
            
            if invalid_payments > 0:
                logger.info(f"payment_types inválidos: {invalid_payments:,}")
                self.df = self.df[~invalid_payment_mask].copy()
        
        # Limpiar store_and_fwd_flag
        if 'store_and_fwd_flag' in self.df.columns:
            valid_flags = ['Y', 'N']
            invalid_flag_mask = ~self.df['store_and_fwd_flag'].isin(valid_flags)
            invalid_flags = invalid_flag_mask.sum()
            
            if invalid_flags > 0:
                logger.info(f"store_and_fwd_flags inválidos: {invalid_flags:,}")
                self.df = self.df[~invalid_flag_mask].copy()
        
        final_count = len(self.df)
        removed_records = initial_count - final_count
        
        logger.info(f"Registros removidos por campos categóricos: {removed_records:,}")
        
        self.preprocessing_stats['categorical_cleaning'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_records': removed_records,
            'removal_percentage': round(removed_records/initial_count*100, 2)
        }
        
        return self.df
    
    def clean_monetary_fields(self) -> pd.DataFrame:
        """
        Limpiar campos monetarios basado en hallazgos del EDA.
        
        Issues identificados:
        - 28.01% registros con discrepancias en total_amount
        - 20,506 tarifas negativas (0.57%)
        - 20,709 totales negativos (0.58%)
        - 135 propinas anómalas en pagos en efectivo
        """
        logger.info("=== LIMPIEZA DE CAMPOS MONETARIOS ===")
        
        initial_count = len(self.df)
        
        # Remover valores negativos en campos que no deberían ser negativos
        positive_fields = ['fare_amount', 'trip_distance', 'total_amount', 'tip_amount', 'tolls_amount']
        
        for field in positive_fields:
            if not field in self.df.columns:
                continue
            negative_mask = self.df[field] < 0
            negative_count = negative_mask.sum()
            
            if negative_count == 0:
                continue
            
            logger.info(f"Valores negativos en {field}: {negative_count:,}")
            self.df = self.df[~negative_mask].copy()
        
        # Filtrar viajes con fare_amount = 0 (problemáticos para cálculos)
        if 'fare_amount' in self.df.columns:
            zero_fare_mask = self.df['fare_amount'] == 0
            zero_fares = zero_fare_mask.sum()
            
            if zero_fares > 0:
                logger.info(f"Viajes con fare_amount = 0: {zero_fares:,}")
                self.df = self.df[~zero_fare_mask].copy()
        
        # Filtrar viajes con distancia 0
        if 'trip_distance' in self.df.columns:
            zero_distance_mask = self.df['trip_distance'] == 0
            zero_distance = zero_distance_mask.sum()
            
            if zero_distance > 0:
                logger.info(f"Viajes con distancia 0: {zero_distance:,}")
                self.df = self.df[~zero_distance_mask].copy()
        
        # Limpiar propinas anómalas en pagos en efectivo
        if 'tip_amount' in self.df.columns and 'payment_type' in self.df.columns:
            cash_tip_mask = (self.df['payment_type'] == 2) & (self.df['tip_amount'] > 0)
            cash_tips = cash_tip_mask.sum()
            
            if cash_tips > 0:
                logger.info(f"Propinas anómalas en efectivo: {cash_tips:,}")
                self.df.loc[cash_tip_mask, 'tip_amount'] = 0
        
        # Aplicar límites razonables a outliers extremos
        if 'fare_amount' in self.df.columns:
            fare_q95 = self.df['fare_amount'].quantile(0.95)
            high_fare_mask = self.df['fare_amount'] > fare_q95 * 3  # 3x el percentil 95
            high_fares = high_fare_mask.sum()
            
            if high_fares > 0:
                logger.info(f"Tarifas extremadamente altas removidas: {high_fares:,}")
                self.df = self.df[~high_fare_mask].copy()
        
        final_count = len(self.df)
        removed_records = initial_count - final_count
        
        logger.info(f"Registros removidos por campos monetarios: {removed_records:,}")
        
        self.preprocessing_stats['monetary_cleaning'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_records': removed_records,
            'removal_percentage': round(removed_records/initial_count*100, 2)
        }
        
        return self.df
    
    def validate_total_amount(self) -> pd.DataFrame:
        """
        Validar y corregir discrepancias en total_amount.
        
        EDA encontró 28.01% registros con discrepancias (diferencia promedio $2.50).
        """
        logger.info("=== VALIDACIÓN DE total_amount ===")
        
        if 'total_amount' not in self.df.columns:
            logger.warning("Campo total_amount no encontrado")
            return self.df
        
        # Calcular total esperado
        base_fields = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge']
        additional_fields = ['congestion_surcharge', 'airport_fee']
        
        # Inicializar con campos base
        calculated_total = pd.Series(0, index=self.df.index)
        
        for field in base_fields:
            if field not in self.df.columns:
                continue
            
            calculated_total += self.df[field].fillna(0)
        
        # Agregar campos adicionales si existen
        for field in additional_fields:
            if field not in self.df.columns:
                continue
            
            calculated_total += self.df[field].fillna(0)
        
        # Calcular diferencias
        diff = abs(self.df['total_amount'] - calculated_total)
        discrepancy_mask = diff > 0.01  # Diferencia mayor a 1 centavo
        discrepancies = discrepancy_mask.sum()
        
        logger.info(f"Registros con discrepancias en total_amount: {discrepancies:,} ({discrepancies/len(self.df)*100:.2f}%)")
        
        if discrepancies > 0:
            logger.info(f"Diferencia promedio: ${diff[discrepancy_mask].mean():.2f}")
            logger.info(f"Diferencia máxima: ${diff.max():.2f}")
            
            # Corregir total_amount con valor calculado
            self.df.loc[discrepancy_mask, 'total_amount'] = calculated_total[discrepancy_mask]
            logger.info("total_amount corregido basado en suma de componentes")
        
        self.preprocessing_stats['total_amount_validation'] = {
            'discrepancies_found': int(discrepancies),
            'discrepancy_percentage': round(discrepancies/len(self.df)*100, 2),
            'avg_difference': round(diff[discrepancy_mask].mean(), 2) if discrepancies > 0 else 0,
            'max_difference': round(diff.max(), 2)
        }
        
        return self.df
    
    def remove_outliers(self, method='iqr', factor=1.5) -> pd.DataFrame:
        """
        Remover outliers extremos usando método IQR.
        
        Args:
            method: Método para detectar outliers ('iqr')
            factor: Factor multiplicador para límites IQR (default 1.5)
        """
        logger.info(f"=== REMOCIÓN DE OUTLIERS ({method.upper()}) ===")
        
        initial_count = len(self.df)
        
        # Campos para análisis de outliers
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
                    logger.info(f"Outliers en {field}: {outliers_count:,} ({outliers_count/len(self.df)*100:.2f}%)")
                    self.df = self.df[~outlier_mask].copy()
                    total_outliers_removed += outliers_count
        
        final_count = len(self.df)
        
        logger.info(f"Total outliers removidos: {total_outliers_removed:,}")
        logger.info(f"Registros restantes: {final_count:,}")
        
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
        Obtener resumen completo del preprocesamiento.
        
        Returns:
            dict: Estadísticas completas del preprocesamiento
        """
        if not self.preprocessing_stats:
            logger.warning("No hay estadísticas de preprocesamiento disponibles")
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
        Ejecutar pipeline completo de limpieza de datos.
        
        Args:
            remove_outliers: Si remover outliers extremos
            outlier_method: Método para detección de outliers
            outlier_factor: Factor para límites de outliers
            
        Returns:
            pd.DataFrame: Datos limpios (sin ingeniería de características)
        """
        logger.info("=== INICIANDO PIPELINE DE LIMPIEZA DE DATOS ===")
        
        # 1. Cargar datos
        self.load_data()
        
        # 2. Validar estructura básica
        self.validate_basic_structure()
        
        # 3. Limpiar campos temporales
        self.clean_datetime_fields()
        
        # 4. Limpiar campos categóricos
        self.clean_categorical_fields()
        
        # 5. Limpiar campos monetarios
        self.clean_monetary_fields()
        
        # 6. Validar total_amount
        self.validate_total_amount()
        
        # 7. Remover outliers si se solicita
        if remove_outliers:
            self.remove_outliers(method=outlier_method, factor=outlier_factor)
        
        # Resumen final
        summary = self.get_preprocessing_summary()
        logger.info("=== LIMPIEZA DE DATOS COMPLETADA ===")
        logger.info(f"Registros originales: {summary['original_shape'][0]:,}")
        logger.info(f"Registros limpios: {summary['total_removed']:,} ({summary['removal_percentage']:.2f}%)")
        logger.info("Datos listos para ingeniería de características")
        
        return self.df
    
    def save_processed_data(self, output_path: str = None) -> str:
        """
        Guardar datos procesados en archivo parquet.
        
        Args:
            output_path: Ruta de salida. Si None, usa PROCESSED_DATA de config.
            
        Returns:
            str: Ruta del archivo guardado
        """
        if self.df is None:
            raise ValueError("No hay datos procesados para guardar. Ejecute el pipeline primero.")
        
        output_path = output_path or PROCESSED_DATA
        
        # Crear directorio si no existe
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar datos
        self.df.to_parquet(output_path, index=False, engine='pyarrow', compression='gzip')
        logger.info(f"Datos procesados guardados en: {output_path}")
        logger.info(f"Tamaño del archivo: {Path(output_path).stat().st_size / 1024**2:.2f} MB")
        
        return output_path


def main():
    """Función principal para ejecutar la limpieza de datos."""
    try:
        # Inicializar preprocesador
        preprocessor = TaxiDataPreprocessor()
        
        # Ejecutar pipeline de limpieza
        cleaned_data = preprocessor.preprocess_full_pipeline(
            remove_outliers=True,
            outlier_method='iqr',
            outlier_factor=1.5
        )
        
        # Guardar datos limpios
        output_path = preprocessor.save_processed_data()
        
        # Mostrar resumen
        summary = preprocessor.get_preprocessing_summary()
        print("\n" + "="*50)
        print("RESUMEN DE LIMPIEZA DE DATOS")
        print("="*50)
        print(f"Registros originales: {summary['original_shape'][0]:,}")
        print(f"Registros limpios: {summary['final_shape'][0]:,}")
        print(f"Registros removidos: {summary['total_removed']:,} ({summary['removal_percentage']:.2f}%)")
        print(f"Archivo de datos limpios: {output_path}")
        print("\n📋 SIGUIENTE PASO:")
        print("   Ejecutar features.py para ingeniería de características")
        
        return cleaned_data, summary
        
    except Exception as e:
        logger.error(f"Error en limpieza de datos: {e}")
        raise


if __name__ == "__main__":
    main()