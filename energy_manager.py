import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pytz


class EnergyManager:
    def __init__(self, data_file="energy_data.json"):
        self.data_file = data_file
        self.devices = []
        self.energy_data = []
        self.tariffs = {"peak": 6.5, "off_peak": 3.2}  # руб/кВт*ч
        self.weather_data = []
        self.load_data()
    
    def load_data(self):
        """Загрузка данных из файла"""
        if os.path.exists(self.data_file):
            with open(self.data_file) as f:
                data = json.load(f)
                self.devices = data.get("devices", [])
                self.energy_data = data.get("energy_data", [])
                self.weather_data = data.get("weather_data", [])
                self.tariffs = data.get("tariffs", self.tariffs)
    
    def save_data(self):
        """Сохранение данных в файл"""
        data = {
            "devices": self.devices,
            "energy_data": self.energy_data,
            "weather_data": self.weather_data,
            "tariffs": self.tariffs
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_device(self, name, power_rating, category, location, schedule=None):
        """Добавление нового устройства"""
        device = {
            "id": len(self.devices) + 1,
            "name": name,
            "power_rating": power_rating,  # Вт
            "category": category,  # lighting, hvac, kitchen, entertainment, etc.
            "location": location,
            "schedule": schedule or {},  # {"start": "08:00", "end": "23:00"}
            "is_smart": False,
            "usage_hours": 0
        }
        self.devices.append(device)
        self.save_data()
        return device

    def add_energy_reading(self, device_id, consumption, timestamp=None):
        """Добавление показания потребления"""
        timestamp = timestamp or datetime.now()
        reading = {
            "device_id": device_id,
            "consumption": consumption,  # кВт*ч
            "timestamp": timestamp.isoformat(),
            "cost": self.calculate_cost(consumption, timestamp)
        }
        self.energy_data.append(reading)
        self.save_data()
        return reading

    def add_weather_data(self, temperature, humidity, timestamp=None):
        """Добавление погодных данных"""
        timestamp = timestamp or datetime.now()
        weather = {
            "temperature": temperature,
            "humidity": humidity,
            "timestamp": timestamp.isoformat()
        }
        self.weather_data.append(weather)
        self.save_data()
        return weather

    def calculate_cost(self, consumption, timestamp):
        """Расчет стоимости потребления с учетом тарифов"""
        dt = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(timestamp)
        hour = dt.hour
        
        # Определяем тарифный период
        tariff_type = "peak" if 7 <= hour < 23 else "off_peak"
        return consumption * self.tariffs[tariff_type]
    
    def get_daily_summary(self, date=None):
        """Получение дневной сводки"""
        date = date or datetime.now().date()
        daily_data = [d for d in self.energy_data if datetime.fromisoformat(d["timestamp"]).date() == date]
        
        if not daily_data:
            return None
        
        total_consumption = sum(d["consumption"] for d in daily_data)
        total_cost = sum(d["cost"] for d in daily_data)
        
        # Группировка по устройствам
        device_consumption = {}
        for d in daily_data:
            device_id = d["device_id"]
            device_name = next((dev["name"] for dev in self.devices if dev["id"] == device_id), "Unknown")
            if device_name not in device_consumption:
                device_consumption[device_name] = 0
            device_consumption[device_name] += d["consumption"]
        
        # Группировка по категориям
        category_consumption = {}
        for d in daily_data:
            device_id = d["device_id"]
            category = next((dev["category"] for dev in self.devices if dev["id"] == device_id), "other")
            if category not in category_consumption:
                category_consumption[category] = 0
            category_consumption[category] += d["consumption"]
        
        return {
            "date": date.isoformat(),
            "total_consumption": total_consumption,
            "total_cost": total_cost,
            "device_breakdown": device_consumption,
            "category_breakdown": category_consumption
        }

    def analyze_usage_patterns(self, days=30):
        """Анализ шаблонов использования"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Фильтрация данных за период
        period_data = [d for d in self.energy_data 
                      if start_date <= datetime.fromisoformat(d["timestamp"]) <= end_date]
        
        if not period_data:
            return {}
        
        # Создание DataFrame для анализа
        df = pd.DataFrame(period_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['date'] = df['timestamp'].dt.date
        
        # Среднее потребление по часам
        hourly_avg = df.groupby('hour')['consumption'].mean()
        
        # Потребление по дням недели
        weekday_avg = df.groupby('day_of_week')['consumption'].mean()
        
        # Потребление по устройствам
        device_consumption = {}
        for device in self.devices:
            device_data = df[df['device_id'] == device['id']]
            if not device_data.empty:
                device_consumption[device['name']] = device_data['consumption'].sum()
        
        return {
            "hourly_avg": hourly_avg.to_dict(),
            "weekday_avg": weekday_avg.to_dict(),
            "device_consumption": device_consumption,
            "total_consumption": df['consumption'].sum(),
            "total_cost": df['cost'].sum()
        }

    def detect_anomalies(self, threshold=2.5):
        """Обнаружение аномалий в потреблении"""
        if not self.energy_data:
            return []
        
        # Создаем временной ряд потребления
        df = pd.DataFrame(self.energy_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Ресемплируем по дням
        daily = df['consumption'].resample('D').sum()
        
        # Вычисляем статистики
        mean = daily.mean()
        std = daily.std()
        
        # Находим аномалии
        anomalies = daily[(daily < mean - threshold * std) | (daily > mean + threshold * std)]
        
        # Форматируем результат
        results = []
        for date, value in anomalies.items():
            results.append({
                "date": date.date().isoformat(),
                "consumption": value,
                "deviation": (value - mean) / std
            })
        
        return results

    def predict_consumption(self, days=7):
        """Прогнозирование потребления на будущее"""
        if not self.energy_data:
            return []
        
        # Создаем временной ряд
        df = pd.DataFrame(self.energy_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Группируем по дням
        daily = df['consumption'].resample('D').sum().reset_index()
        daily['day_of_week'] = daily['timestamp'].dt.dayofweek
        daily['day'] = daily['timestamp'].dt.day
        daily['month'] = daily['timestamp'].dt.month
        
        # Подготовка данных для модели
        X = daily[['day_of_week', 'day', 'month']]
        y = daily['consumption']
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Обучение модели
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Точность прогноза: MAE = {mae:.3f} кВт*ч")
        
        # Прогноз на будущее
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days+1)]
        future_data = pd.DataFrame({
            'timestamp': future_dates,
            'day_of_week': [d.weekday() for d in future_dates],
            'day': [d.day for d in future_dates],
            'month': [d.month for d in future_dates]
        })
        
        predictions = model.predict(future_data[['day_of_week', 'day', 'month']])
        
        # Форматируем результат
        forecast = []
        for date, pred in zip(future_dates, predictions):
            forecast.append({
                "date": date.date().isoformat(),
                "predicted_consumption": pred,
                "predicted_cost": self.calculate_cost(pred, date)
            })
        
        return forecast