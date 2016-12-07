import seaborn as sns
import pandas as pd

flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
print flights
