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