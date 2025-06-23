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

    def get_energy_saving_tips(self):
        """Получение советов по экономии энергии"""
        tips = []
        patterns = self.analyze_usage_patterns(30)
        
        # Анализ пикового потребления
        if "hourly_avg" in patterns:
            peak_hours = [hour for hour, cons in patterns["hourly_avg"].items() if cons > 1.5 * np.mean(list(patterns["hourly_avg"].values()))]
            if peak_hours:
                tips.append(f"Снизьте потребление в пиковые часы: {', '.join(str(h) for h in sorted(peak_hours))}:00")
        
        # Анализ устройств с высоким потреблением
        if "device_consumption" in patterns:
            top_consumers = sorted(patterns["device_consumption"].items(), key=lambda x: x[1], reverse=True)[:3]
            for device, consumption in top_consumers:
                if consumption > 5:  # кВт*ч за месяц
                    tips.append(f"Устройство '{device}' потребляет много энергии. Проверьте его настройки.")
        
        # Общие советы
        tips.append("Используйте LED освещение вместо ламп накаливания")
        tips.append("Установите программируемый термостат для отопления/охлаждения")
        tips.append("Выключайте устройства из розетки, когда они не используются")
        
        return tips or ["Ваше энергопотребление выглядит эффективным!"]

    def plot_energy_usage(self, days=7):
        """Визуализация энергопотребления"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Фильтрация данных
        period_data = [d for d in self.energy_data 
                      if start_date <= datetime.fromisoformat(d["timestamp"]) <= end_date]
        
        if not period_data:
            print("Нет данных для визуализации")
            return
        
        # Создаем DataFrame
        df = pd.DataFrame(period_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Группируем по дням/часам
        daily = df['consumption'].resample('D').sum()
        hourly = df['consumption'].resample('H').sum()
        
        # Создаем графики
        plt.figure(figsize=(15, 10))
        
        # Суточное потребление
        plt.subplot(2, 1, 1)
        daily.plot(kind='bar', color='skyblue')
        plt.title('Суточное потребление энергии')
        plt.xlabel('Дата')
        plt.ylabel('кВт*ч')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Почасовое потребление
        plt.subplot(2, 1, 2)
        hourly.plot(kind='line', marker='o', color='green')
        plt.title('Почасовое потребление энергии')
        plt.xlabel('Время')
        plt.ylabel('кВт*ч')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

    def plot_cost_analysis(self, days=30):
        """Анализ стоимости энергопотребления"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Фильтрация данных
        period_data = [d for d in self.energy_data 
                      if start_date <= datetime.fromisoformat(d["timestamp"]) <= end_date]
        
        if not period_data:
            print("Нет данных для визуализации")
            return
        
        # Создаем DataFrame
        df = pd.DataFrame(period_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['tariff'] = df['hour'].apply(lambda h: "peak" if 7 <= h < 23 else "off_peak")
        
        # Группируем по тарифам
        cost_by_tariff = df.groupby('tariff')['cost'].sum()
        consumption_by_tariff = df.groupby('tariff')['consumption'].sum()
        
        # Создаем графики
        plt.figure(figsize=(12, 8))
        
        # Распределение стоимости по тарифам
        plt.subplot(2, 2, 1)
        cost_by_tariff.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
        plt.title('Распределение стоимости по тарифам')
        plt.ylabel('')
        
        # Распределение потребления по тарифам
        plt.subplot(2, 2, 2)
        consumption_by_tariff.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
        plt.title('Распределение потребления по тарифам')
        plt.ylabel('')
        
        # Сравнение стоимости и потребления
        plt.subplot(2, 1, 2)
        width = 0.35
        x = np.arange(len(cost_by_tariff))
        
        plt.bar(x - width/2, consumption_by_tariff, width, label='Потребление (кВт*ч)', color='skyblue')
        plt.bar(x + width/2, cost_by_tariff, width, label='Стоимость (руб)', color='salmon')
        
        plt.title('Сравнение потребления и стоимости')
        plt.xlabel('Тарифный период')
        plt.xticks(x, cost_by_tariff.index)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()


# Генератор демо-данных
def generate_demo_data(manager):
    """Создание демонстрационных данных"""
    # Добавляем устройства
    devices = [
        manager.add_device("Холодильник", 150, "kitchen", "Кухня", {"start": "00:00", "end": "23:59"}),
        manager.add_device("Кондиционер", 1500, "hvac", "Гостиная", {"start": "12:00", "end": "22:00"}),
        manager.add_device("Стиральная машина", 2000, "laundry", "Ванная", {"start": "08:00", "end": "23:00"}),
        manager.add_device("Телевизор", 120, "entertainment", "Гостиная", {"start": "18:00", "end": "23:00"}),
        manager.add_device("Освещение", 300, "lighting", "Весь дом", {"start": "17:00", "end": "23:00"})
    ]
    
    # Генерируем показания потребления
    now = datetime.now()
    for i in range(30):
        date = now - timedelta(days=30-i)
        for device in devices:
            # Базовое потребление
            base_consumption = device["power_rating"] / 1000 * np.random.uniform(0.5, 1.5)
            
            # Учитываем расписание
            start_hour = int(device["schedule"]["start"].split(":")[0])
            end_hour = int(device["schedule"]["end"].split(":")[0])
            
            for hour in range(24):
                if start_hour <= hour <= end_hour:
                    # Потребление в рабочее время
                    consumption = base_consumption * np.random.uniform(0.8, 1.2)
                else:
                    # Потребление в нерабочее время (может быть не нулевым для некоторых устройств)
                    consumption = base_consumption * np.random.uniform(0.1, 0.3)
                
                timestamp = datetime(date.year, date.month, date.day, hour)
                manager.add_energy_reading(device["id"], consumption, timestamp)
    
    # Генерируем погодные данные
    for i in range(30):
        date = now - timedelta(days=30-i)
        temp = 15 + 10 * np.sin(i/5) + np.random.normal(0, 3)
        humidity = 50 + 20 * np.cos(i/7) + np.random.normal(0, 10)
        timestamp = datetime(date.year, date.month, date.day, 12)
        manager.add_weather_data(temp, humidity, timestamp)
