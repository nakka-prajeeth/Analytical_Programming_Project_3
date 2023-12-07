import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

class MonthlyTrendAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the MonthlyTrendAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def convertToDatetime(self):
        """
        Convert 'OCCUR_DATE' to datetime.

        Returns:
        None
        """
        self.shootingData['OCCUR_DATE'] = pd.to_datetime(self.shootingData['OCCUR_DATE'])
        print("Datetime")

    def extractYearMonth(self):
        """
        Extract the year and month for trend analysis.

        Returns:
        None
        """
        self.shootingData['YEAR'] = self.shootingData['OCCUR_DATE'].dt.year
        self.shootingData['MONTH'] = self.shootingData['OCCUR_DATE'].dt.month

    def analyzeMonthlyTrends(self):
        """
        Analyze monthly trends by grouping data.

        Returns:
        DataFrame: Grouped data with incident counts.
        """
        monthlyTrends = self.shootingData.groupby(['YEAR', 'MONTH']).size().reset_index(name='INCIDENT_COUNTS')
        return monthlyTrends

    def visualizeHeatmap(self, trendsData):
        """
        Visualize a heatmap of monthly incident trends.

        Parameters:
        - trendsData (DataFrame): Grouped data with incident counts.

        Returns:
        None
        """
        # Pivot the data for visualization
        monthlyTrendsPivot = trendsData.pivot('MONTH', 'YEAR', 'INCIDENT_COUNTS')


class ShootingOutcomeAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the ShootingOutcomeAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def determineOutcome(self):
        """
        Determine the outcome of shooting incidents (Fatal or Non-Fatal).

        Returns:
        None
        """
        self.shooting_data['OUTCOME'] = self.shooting_data['STATISTICAL_MURDER_FLAG'].map({True: 'Fatal', False: 'Non-Fatal'})

class ShootingOutcomeAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the ShootingOutcomeAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def analyzeOutcomeCounts(self):
        """
        Analyze the counts of Fatal and Non-Fatal shooting incidents.

        Returns:
        Series: Counts of each outcome.
        """
        outcomeCounts = self.shootingData['OUTCOME'].value_counts()
        return outcomeCounts

    def visualizeOutcomeBarplot(self, outcomeCounts):
        """
        Visualize a barplot of shooting incident outcomes.

        Parameters:
        - outcomeCounts (Series): Counts of each outcome.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x=outcomeCounts.index, y=outcomeCounts.values, palette='bright')
        plt.title('Outcomes of Shooting Incidents')
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        plt.show()


class ShootingLocationTimeAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the ShootingLocationTimeAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def convertOccurrenceTime(self):
        """
        Convert 'OCCUR_TIME' to datetime and extract the hour.

        Returns:
        None
        """
        self.shootingData['OCCUR_TIME'] = pd.to_datetime(self.shootingData['OCCUR_TIME'], format='%H:%M:%S').dt.hour

    def categorizeTimeOfDay(self, hour):
        """
        Categorize the time of day based on the hour.

        Parameters:
        - hour (int): The hour of the day.

        Returns:
        str: Categorized time of day.
        """
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

class ShootingLocationTimeAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the ShootingLocationTimeAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def convertOccurrenceTime(self):
        """
        Convert 'OCCUR_TIME' to datetime and extract the hour.

        Returns:
        None
        """
        self.shootingData['OCCUR_TIME'] = pd.to_datetime(self.shootingData['OCCUR_TIME'], format='%H:%M:%S').dt.hour

    def categorizeTimeOfDay(self, hour):
        """
        Categorize the time of day based on the hour.

        Parameters:
        - hour (int): The hour of the day.

        Returns:
        str: Categorized time of day.
        """
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    def applyTimeOfDayCategorization(self):
        """
        Apply the function to categorize time of day.

        Returns:
        None
        """
        self.shootingData['TIME_OF_DAY'] = self.shootingData['OCCUR_TIME'].apply(self.categorizeTimeOfDay)

    def analyzeLocationTimeDistribution(self):
        """
        Analyze the distribution of shooting incidents by location and time of day.

        Returns:
        DataFrame: Crosstab of location and time distribution.
        """
        locationTimeDistribution = pd.crosstab(self.shootingData['LOCATION_DESC'], self.shootingData['TIME_OF_DAY'])
        return locationTimeDistribution

    def visualizeHeatmap(self, locationTimeDistribution):
        """
        Visualize a heatmap of shooting incidents by location and time of day.

        Parameters:
        - locationTimeDistribution (DataFrame): Crosstab of location and time distribution.

        Returns:
        None
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(locationTimeDistribution, annot=True, fmt='d', cmap='viridis')
        plt.title('Shooting Incidents by Location and Time of Day')
        plt.ylabel('Location Type')
        plt.xlabel('Time of Day')
        plt.show()


class ShootingLocationAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the ShootingLocationAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def analyzeLocationDistribution(self):
        """
        Analyze the distribution of shooting incidents by location.

        Returns:
        Series: Value counts of shooting incident locations.
        """
        locationCounts = self.shootingData['LOC_OF_OCCUR_DESC'].value_counts()
        return locationCounts

    def visualizeLocationDistribution(self, locationCounts):
        """
        Visualize the distribution of shooting incidents by location.

        Parameters:
        - locationCounts (Series): Value counts of shooting incident locations.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x=locationCounts.index, y=locationCounts.values, palette='coolwarm')
        plt.title('Shooting Incidents: Inside vs Outside')
        plt.xlabel('Location of Occurrence')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


class ShootingTimeAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the ShootingTimeAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def preprocessOccurTime(self):
        """
        Preprocess the 'OCCUR_TIME' column to ensure consistent formatting.

        Returns:
        None
        """
        self.shootingData['OCCUR_TIME'] = self.shootingData['OCCUR_TIME'].apply(
            lambda x: f'{x}:00:00' if len(str(x)) == 2 else x
        )

    def categorizeAndVisualizeTimeDistribution(self):
        """
        Categorize and visualize the distribution of shooting incidents by time category.

        Returns:
        None
        """
        # Convert 'OCCUR_TIME' to datetime and extract the hour
        self.shootingData['OCCUR_TIME'] = pd.to_datetime(self.shootingData['OCCUR_TIME'], format='%H:%M:%S').dt.hour

        # Categorize time of day
        self.shootingData['TIME_CATEGORY'] = pd.cut(
            self.shootingData['OCCUR_TIME'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            right=False
        )

        # Visualize the distribution of shooting incidents by time category
        plt.figure(figsize=(10, 6))
        sns.countplot(x='TIME_CATEGORY', data=self.shootingData, palette='Set2')
        plt.title('Distribution of Shooting Incidents by Time Category')
        plt.xlabel('Time Category')
        plt.ylabel('Count')
        plt.show()


class ShootingJurisdictionAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the ShootingJurisdictionAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def visualizeJurisdictionDistribution(self):
        """
        Visualize the distribution of shooting incidents by jurisdiction.

        Returns:
        None
        """
        # Count occurrences of each jurisdiction code
        jurisdictionCounts = self.shootingData['JURISDICTION_CODE'].value_counts()

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x=jurisdictionCounts.index, y=jurisdictionCounts.values, palette='cubehelix')
        plt.title('Shooting Incidents by Jurisdiction')
        plt.xlabel('Jurisdiction Code')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=45)
        plt.show()


class TopPrecinctsAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the TopPrecinctsAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def visualizeTopPrecincts(self, topN=20):
        """
        Visualize the top N precincts with the highest number of shooting incidents.

        Parameters:
        - topN (int): Number of top precincts to display.

        Returns:
        None
        """
        # Count occurrences of each precinct
        precinctCounts = self.shootingData['PRECINCT'].value_counts().head(topN)

        # Plotting
        plt.figure(figsize=(12, 6))
        sns.barplot(x=precinctCounts.index, y=precinctCounts.values, palette='rocket')
        plt.title(f'Top {topN} Precincts with Highest Number of Shooting Incidents')
        plt.xlabel('Precinct')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=45)
        plt.show()


class DayOfWeekAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the DayOfWeekAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)
        # Convert 'OCCUR_DATE' to datetime type
        self.shootingData['OCCUR_DATE'] = pd.to_datetime(self.shootingData['OCCUR_DATE'], errors='coerce')

    def visualizeShootingDistributionByDay(self):
        """
        Visualize the distribution of shooting incidents by day of the week.

        Returns:
        None
        """
        # Extract the day of the week from the 'OCCUR_DATE' column
        self.shootingData['DAY_OF_WEEK'] = self.shootingData['OCCUR_DATE'].dt.day_name()

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.countplot(
            x='DAY_OF_WEEK',
            data=self.shootingData,
            order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            palette='muted'
        )
        plt.title('Distribution of Shooting Incidents by Day of the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Incidents')
        plt.show()


class LocationBoroughAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the LocationBoroughAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def visualizeShootingDistributionByLocationAndBorough(self):
        """
        Visualize the distribution of shooting incidents by location description and borough.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.countplot(
            x='LOCATION_DESC',
            hue='BORO',
            data=self.shootingData,
            palette='husl'
        )
        plt.title('Distribution of Shooting Incidents by Location and Borough')
        plt.xlabel('Location Description')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=90)
        plt.legend(title='Borough')
        plt.show()


class DayOfMonthAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the DayOfMonthAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)
        # Convert 'OCCUR_DATE' to datetime
        self.shootingData['OCCUR_DATE'] = pd.to_datetime(self.shootingData['OCCUR_DATE'], errors='coerce')

    def visualizeShootingDistributionByDayOfMonth(self):
        """
        Visualize the distribution of shooting incidents by day of the month.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.countplot(
            x=self.shootingData['OCCUR_DATE'].dt.day,
            palette='muted'
        )
        plt.title('Distribution of Shooting Incidents by Day of the Month')
        plt.xlabel('Day of the Month')
        plt.ylabel('Number of Incidents')
        plt.show()


class BoroughJurisdictionAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the BoroughJurisdictionAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def visualizeShootingDistributionByBoroughAndJurisdiction(self):
        """
        Visualize the distribution of shooting incidents by borough and jurisdiction code.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.countplot(x='BORO', hue='JURISDICTION_CODE', data=self.shootingData, palette='deep')
        plt.title('Distribution of Shooting Incidents by Borough and Jurisdiction Code')
        plt.xlabel('Borough')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Jurisdiction Code')
        plt.show()

class VictimAgeSexAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the VictimAgeSexAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def visualizeShootingDistributionByAgeAndSex(self):
        """
        Visualize the distribution of shooting incidents by victim age group and sex.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.countplot(x='VIC_AGE_GROUP', hue='VIC_SEX', data=self.shootingData, palette='coolwarm')
        plt.title('Shooting Incidents by Victim Age Group and Sex')
        plt.xlabel('Victim Age Group')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Victim Sex')
        plt.xticks(rotation=90)
        plt.show()


class BoroughSuspectRaceAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the BoroughSuspectRaceAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def visualizeShootingDistributionByBoroughAndRace(self):
        """
        Visualize shooting incidents by borough and suspect race using a stacked bar plot.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.countplot(x='BORO', hue='PERP_RACE', data=self.shootingData, palette='Set2')
        plt.title('Shooting Incidents by Borough and Suspect Race')
        plt.xlabel('Borough')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Suspect Race')
        plt.show()

class TimeSeriesAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the TimeSeriesAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)
        self.shootingData['OCCUR_DATE'] = pd.to_datetime(self.shootingData['OCCUR_DATE'])

    def visualizeMonthlyTrendOfShootingIncidents(self):
        """
        Visualize the monthly trend of shooting incidents using a time series plot.

        Returns:
        None
        """
        # Resample data to get monthly counts
        monthlyCounts = self.shootingData.resample('M', on='OCCUR_DATE').size()

        # Plotting
        plt.figure(figsize=(14, 8))
        monthlyCounts.plot(marker='o', linestyle='-', color='purple')
        plt.title('Monthly Trend of Shooting Incidents')
        plt.xlabel('Date')
        plt.ylabel('Number of Incidents')
        plt.show()

class BoroughLocationAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the BoroughLocationAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def visualizeShootingDistributionByBoroughAndLocation(self):
        """
        Visualize shooting incidents by borough and location description using a count plot.

        Returns:
        None
        """
        plt.figure(figsize=(14, 8))
        sns.countplot(x='BORO', hue='LOCATION_DESC', data=self.shootingData, palette='Pastel1')
        plt.title('Shooting Incidents by Borough and Location Description')
        plt.xlabel('Borough')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Location Description', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


class PrecinctJurisdictionAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the PrecinctJurisdictionAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def visualizeShootingDistributionByPrecinctAndJurisdiction(self):
        """
        Visualize shooting incidents by precinct and jurisdiction code using a count plot.

        Returns:
        None
        """
        plt.figure(figsize=(14, 8))
        sns.countplot(x='PRECINCT', hue='JURISDICTION_CODE', data=self.shootingData, palette='viridis')
        plt.title('Shooting Incidents by Precinct and Jurisdiction Code')
        plt.xlabel('Precinct')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Jurisdiction Code', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


class DayBoroughAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the DayBoroughAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def visualizeShootingDistributionByDayAndBorough(self):
        """
        Visualize shooting incidents by day and borough using a stacked bar plot.

        Returns:
        None
        """
        # Convert 'OCCUR_DATE' to datetime
        self.shootingData['OCCUR_DATE'] = pd.to_datetime(self.shootingData['OCCUR_DATE'])

        # Extract the day of the week from the 'OCCUR_DATE' column
        self.shootingData['DAY'] = self.shootingData['OCCUR_DATE'].dt.day_name()

        # Group by day and borough and get counts
        dayBoroughCounts = self.shootingData.groupby(['DAY', 'BORO']).size().unstack()

        # Plotting
        plt.figure(figsize=(14, 8))
        dayBoroughCounts.plot(kind='bar', stacked=True, cmap='tab10')
        plt.title('Shooting Incidents by Day and Borough')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Borough', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


class AgeGenderAnalysis:
    def __init__(self, dataUrl):
        """
        Initialize the AgeGenderAnalysis class.

        Parameters:
        - dataUrl (str): URL to the CSV file containing shooting incident data.
        """
        self.shootingData = pd.read_csv(dataUrl)

    def visualizeShootingDistributionByAgeAndGender(self):
        """
        Visualize shooting incidents by victim age group and gender using a count plot.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(14, 8))
        sns.countplot(x='VIC_AGE_GROUP', hue='VIC_SEX', data=self.shootingData, palette='Set2')
        plt.title('Shooting Incidents by Victim Age Group and Gender')
        plt.xlabel('Victim Age Group')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Victim Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
