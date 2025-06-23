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
    