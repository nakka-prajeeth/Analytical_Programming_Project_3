import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CrimeDataAnalyzer:
    def __init__(self, dataUrl):
        """
        Initialize the CrimeDataAnalyzer instance.

        Parameters:
        - dataUrl (str): The URL to the crime data CSV file.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def cleanData(self):
        """
        Clean the crime data by converting 'OCCUR_DATE' to datetime type,
        filtering data for years up to 2021, and extracting the year.
        """
        # Convert 'OCCUR_DATE' to datetime type
        self.shootingData['OCCUR_DATE'] = pd.to_datetime(self.shootingData['OCCUR_DATE'], errors='coerce')

        # Filtering data for years up to 2021
        self.shootingData = self.shootingData[self.shootingData['OCCUR_DATE'].dt.year <= 2021]

        # Extract the year from the 'OCCUR_DATE' column
        self.shootingData['YEAR'] = self.shootingData['OCCUR_DATE'].dt.year

    def plotCrimeDistribution(self):
        """
        Plot the distribution of crimes per year.
        """
        # Plotting the distribution of crimes per year
        plt.figure(figsize=(6, 5))
        self.shootingData['YEAR'].value_counts().sort_index().plot(kind='bar', color='green')
        plt.title('Crime Distribution Over the Years')
        plt.xlabel('Year')
        plt.ylabel('Number of Crimes')
        plt.show()

class ShootingAnalysis:
    def __init__(self, shootingData):
            """
            Initialize the ShootingAnalysis class.

            Parameters:
            - shootingData (pd.DataFrame): DataFrame containing shooting incident data.
            """
            self.shootingData = shootingData

    def plotPrecinctDistribution(self):
            """
            Plot the distribution of shooting incidents by precinct.

            Returns:
            None
            """
            # Creating a count plot to visualize shooting incidents by precinct
            plt.figure(figsize=(12, 6))
            sns.countplot(x='PRECINCT', data=self.shootingData, palette='viridis')
            plt.title('Distribution of Shooting Incidents by Precinct')
            plt.xlabel('Precinct')
            plt.ylabel('Number of Incidents')
            plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
            plt.show()

    def mapRacePopulation(self, racePopulation):
    """
    Map race population data to the shooting data based on perpetrator race.

    Parameters:
    - racePopulation (dict): Dictionary mapping perpetrator races to their respective populations.

    Returns:
    None
    """
    self.shootingData['PERP_RACE_POPULATION'] = self.shootingData['PERP_RACE'].map(racePopulation)

def plotRaceDistribution(self):
    """
    Plot the distribution of shooting incidents by perpetrator race.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='PERP_RACE', data=self.shootingData, palette='Set2')
    plt.title('Distribution of Shooting Incidents by Perpetrator Race')
    plt.xlabel('Perpetrator Race')
    plt.ylabel('Count')
    plt.xticks(rotation=90, ha='right')
    plt.show()

class AgeGroupAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the AgeGroupAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def cleanAgeGroupData(self):
        """
        Clean 'PERP_AGE_GROUP' data by replacing various null values with NaN.

        Returns:
        None
        """
        nullValues = ['nan', 'UNKNOWN', '(null)']
        self.shootingData['PERP_AGE_GROUP'].replace(nullValues, np.nan, inplace=True)

    def identifyNumericAnomalies(self):
        """
        Identify rows with numeric anomalies in 'PERP_AGE_GROUP' (e.g., '940', '224', '1020').

        Returns:
        DataFrame: Rows with numeric anomalies.
        """
        anomalousRows = self.shootingData[self.shootingData['PERP_AGE_GROUP'].isin(['940', '224', '1020'])]
        return anomalousRows

    def handleNumericAnomalies(self):
        """
        Handle numeric anomalies in 'PERP_AGE_GROUP' by replacing specific values with NaN.

        Returns:
        None
        """
        self.shootingData['PERP_AGE_GROUP'].replace(['940', '224', '1020'], np.nan, inplace=True)

    def plotAgeGroupDistribution(self):
        """
        Plot the distribution of shooting incidents by perpetrator age group.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x='PERP_AGE_GROUP', data=self.shootingData, palette='Set2')
        plt.title('Distribution of Shooting Incidents by Perpetrator Age Group')
        plt.xlabel('Perpetrator Age Group')
        plt.ylabel('Count')
        plt.show()
