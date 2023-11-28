import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CrimeDataAnalyzer:
    def __init__(self, data_url):
        """
        Initialize the CrimeDataAnalyzer instance.

        Parameters:
        - data_url (str): The URL to the crime data CSV file.
        """
        self.shooting_data = pd.read_csv(data_url)

    def clean_data(self):
        """
        Clean the crime data by converting 'OCCUR_DATE' to datetime type,
        filtering data for years up to 2021, and extracting the year.
        """
        # Convert 'OCCUR_DATE' to datetime type
        self.shooting_data['OCCUR_DATE'] = pd.to_datetime(self.shooting_data['OCCUR_DATE'], errors='coerce')

        # Filtering data for years up to 2021
        self.shooting_data = self.shooting_data[self.shooting_data['OCCUR_DATE'].dt.year <= 2021]

        # Extract the year from the 'OCCUR_DATE' column
        self.shooting_data['YEAR'] = self.shooting_data['OCCUR_DATE'].dt.year

    def plot_crime_distribution(self):
        """
        Plot the distribution of crimes per year.
        """
        # Plotting the distribution of crimes per year
        plt.figure(figsize=(6, 5))
        self.shooting_data['YEAR'].value_counts().sort_index().plot(kind='bar', color='green')
        plt.title('Crime Distribution Over the Years')
        plt.xlabel('Year')
        plt.ylabel('Number of Crimes')
        plt.show()

class ShootingAnalysis:
    def __init__(self, shooting_data):
            """
            Initialize the ShootingAnalysis class.

            Parameters:
            - shooting_data (pd.DataFrame): DataFrame containing shooting incident data.
            """
            self.shooting_data = shooting_data

    def plot_precinct_distribution(self):
            """
            Plot the distribution of shooting incidents by precinct.

            Returns:
            None
            """
            # Creating a count plot to visualize shooting incidents by precinct
            plt.figure(figsize=(12, 6))
            sns.countplot(x='PRECINCT', data=self.shooting_data, palette='viridis')
            plt.title('Distribution of Shooting Incidents by Precinct')
            plt.xlabel('Precinct')
            plt.ylabel('Number of Incidents')
            plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
            plt.show()

    def map_race_population(self, race_population):
        """
        Map race population data to the shooting data based on perpetrator race.

        Parameters:
        - race_population (dict): Dictionary mapping perpetrator races to their respective populations.

        Returns:
        None
        """
        self.shooting_data['PERP_RACE_POPULATION'] = self.shooting_data['PERP_RACE'].map(race_population)

    def plot_race_distribution(self):
        """
        Plot the distribution of shooting incidents by perpetrator race.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x='PERP_RACE', data=self.shooting_data, palette='Set2')
        plt.title('Distribution of Shooting Incidents by Perpetrator Race')
        plt.xlabel('Perpetrator Race')
        plt.ylabel('Count')
        plt.xticks(rotation=90, ha='right')
        plt.show()

class AgeGroupAnalysis:
    def __init__(self, data_url):
        """
        Initialize the AgeGroupAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def clean_age_group_data(self):
        """
        Clean 'PERP_AGE_GROUP' data by replacing various null values with NaN.

        Returns:
        None
        """
        null_values = ['nan', 'UNKNOWN', '(null)']
        self.shooting_data['PERP_AGE_GROUP'].replace(null_values, np.nan, inplace=True)

    def identify_numeric_anomalies(self):
        """
        Identify rows with numeric anomalies in 'PERP_AGE_GROUP' (e.g., '940', '224', '1020').

        Returns:
        DataFrame: Rows with numeric anomalies.
        """
        anomalous_rows = self.shooting_data[self.shooting_data['PERP_AGE_GROUP'].isin(['940', '224', '1020'])]
        return anomalous_rows

    def handle_numeric_anomalies(self):
        """
        Handle numeric anomalies in 'PERP_AGE_GROUP' by replacing specific values with NaN.

        Returns:
        None
        """
        self.shooting_data['PERP_AGE_GROUP'].replace(['940', '224', '1020'], np.nan, inplace=True)

    def plot_age_group_distribution(self):
        """
        Plot the distribution of shooting incidents by perpetrator age group.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x='PERP_AGE_GROUP', data=self.shooting_data, palette='Set2')
        plt.title('Distribution of Shooting Incidents by Perpetrator Age Group')
        plt.xlabel('Perpetrator Age Group')
        plt.ylabel('Count')
        plt.show()


