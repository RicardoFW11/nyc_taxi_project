import numpy as np
import random

def calculate_reliability(fare_amount):
    """Calcula un porcentaje de confianza basado en la complejidad del viaje."""
    if fare_amount < 15:
        return 92
    elif fare_amount < 40:
        return 85
    else:
        return 78

def simulate_uber_fare(distance_miles, duration_minutes, pickup_datetime):
    """Motor Analítico: Simulación de Tarifa Uber basada en reglas de negocio."""
    base_fare = 2.55
    cost_per_minute = 0.35
    cost_per_mile = 2.15
    
    uber_subtotal = base_fare + (duration_minutes * cost_per_minute) + (distance_miles * cost_per_mile)
    
    hour = pickup_datetime.hour
    is_rush_hour = 7 <= hour <= 9 or 16 <= hour <= 19
    
    surge_multiplier = 1.0
    if is_rush_hour:
        surge_multiplier = round(random.uniform(1.2, 1.8), 2)
    
    final_fare = uber_subtotal * surge_multiplier
    return final_fare, surge_multiplier

def calculate_advanced_metrics(y_true, y_pred):
    """
    Calcula métricas de distribución de errores para el informe del mentor.
    """
    residuals = y_pred - y_true
    abs_error_pct = np.abs(residuals / y_true) * 100
    
    # El 80% de las ejecuciones tienen un error menor a este valor:
    reliability_80 = np.percentile(abs_error_pct, 80) 
    
    # Conteo de equivocaciones (ej. error mayor a $3 USD)
    major_errors = np.sum(np.abs(residuals) > 3.0)
    
    return {
        "reliability_80_percentile": round(reliability_80, 2),
        "total_major_errors": int(major_errors),
        "mean_absolute_error": round(np.mean(np.abs(residuals)), 2)
    }