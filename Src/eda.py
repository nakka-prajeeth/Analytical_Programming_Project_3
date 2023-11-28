import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

class MonthlyTrendAnalysis:
    def __init__(self, data_url):
        """
        Initialize the MonthlyTrendAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def convert_to_datetime(self):
        """
        Convert 'OCCUR_DATE' to datetime.

        Returns:
        None
        """
        self.shooting_data['OCCUR_DATE'] = pd.to_datetime(self.shooting_data['OCCUR_DATE'])

    def extract_year_month(self):
        """
        Extract the year and month for trend analysis.

        Returns:
        None
        """
        self.shooting_data['YEAR'] = self.shooting_data['OCCUR_DATE'].dt.year
        self.shooting_data['MONTH'] = self.shooting_data['OCCUR_DATE'].dt.month

    def analyze_monthly_trends(self):
        """
        Analyze monthly trends by grouping data.

        Returns:
        DataFrame: Grouped data with incident counts.
        """
        monthly_trends = self.shooting_data.groupby(['YEAR', 'MONTH']).size().reset_index(name='INCIDENT_COUNTS')
        return monthly_trends

    def visualize_heatmap(self, trends_data):
        """
        Visualize a heatmap of monthly incident trends.

        Parameters:
        - trends_data (DataFrame): Grouped data with incident counts.

        Returns:
        None
        """
        # Pivot the data for visualization
        monthly_trends_pivot = trends_data.pivot('MONTH', 'YEAR', 'INCIDENT_COUNTS')
class ShootingOutcomeAnalysis:
    def __init__(self, data_url):
        """
        Initialize the ShootingOutcomeAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def determine_outcome(self):
        """
        Determine the outcome of shooting incidents (Fatal or Non-Fatal).

        Returns:
        None
        """
        self.shooting_data['OUTCOME'] = self.shooting_data['STATISTICAL_MURDER_FLAG'].map({True: 'Fatal', False: 'Non-Fatal'})

    def analyze_outcome_counts(self):
        """
        Analyze the counts of Fatal and Non-Fatal shooting incidents.

        Returns:
        Series: Counts of each outcome.
        """
        outcome_counts = self.shooting_data['OUTCOME'].value_counts()
        return outcome_counts

    def visualize_outcome_barplot(self, outcome_counts):
        """
        Visualize a barplot of shooting incident outcomes.

        Parameters:
        - outcome_counts (Series): Counts of each outcome.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x=outcome_counts.index, y=outcome_counts.values, palette='bright')
        plt.title('Outcomes of Shooting Incidents')
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        plt.show()

class ShootingLocationTimeAnalysis:
    def __init__(self, data_url):
        """
        Initialize the ShootingLocationTimeAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def convert_occurrence_time(self):
        """
        Convert 'OCCUR_TIME' to datetime and extract the hour.

        Returns:
        None
        """
        self.shooting_data['OCCUR_TIME'] = pd.to_datetime(self.shooting_data['OCCUR_TIME'], format='%H:%M:%S').dt.hour

    def categorize_time_of_day(self, hour):
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

    def apply_time_of_day_categorization(self):
        """
        Apply the function to categorize time of day.

        Returns:
        None
        """
        self.shooting_data['TIME_OF_DAY'] = self.shooting_data['OCCUR_TIME'].apply(self.categorize_time_of_day)

    def analyze_location_time_distribution(self):
        """
        Analyze the distribution of shooting incidents by location and time of day.

        Returns:
        DataFrame: Crosstab of location and time distribution.
        """
        location_time_distribution = pd.crosstab(self.shooting_data['LOCATION_DESC'], self.shooting_data['TIME_OF_DAY'])
        return location_time_distribution

    def visualize_heatmap(self, location_time_distribution):
        """
        Visualize a heatmap of shooting incidents by location and time of day.

        Parameters:
        - location_time_distribution (DataFrame): Crosstab of location and time distribution.

        Returns:
        None
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(location_time_distribution, annot=True, fmt='d', cmap='viridis')
        plt.title('Shooting Incidents by Location and Time of Day')
        plt.ylabel('Location Type')
        plt.xlabel('Time of Day')
        plt.show()

class ShootingLocationAnalysis:
    def __init__(self, data_url):
        """
        Initialize the ShootingLocationAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def analyze_location_distribution(self):
        """
        Analyze the distribution of shooting incidents by location.

        Returns:
        Series: Value counts of shooting incident locations.
        """
        location_counts = self.shooting_data['LOC_OF_OCCUR_DESC'].value_counts()
        return location_counts

    def visualize_location_distribution(self, location_counts):
        """
        Visualize the distribution of shooting incidents by location.

        Parameters:
        - location_counts (Series): Value counts of shooting incident locations.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x=location_counts.index, y=location_counts.values, palette='coolwarm')
        plt.title('Shooting Incidents: Inside vs Outside')
        plt.xlabel('Location of Occurrence')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


class ShootingTimeAnalysis:
    def __init__(self, data_url):
        """
        Initialize the ShootingTimeAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def preprocess_occur_time(self):
        """
        Preprocess the 'OCCUR_TIME' column to ensure consistent formatting.

        Returns:
        None
        """
        self.shooting_data['OCCUR_TIME'] = self.shooting_data['OCCUR_TIME'].apply(
            lambda x: f'{x}:00:00' if len(str(x)) == 2 else x
        )

    def categorize_and_visualize_time_distribution(self):
        """
        Categorize and visualize the distribution of shooting incidents by time category.

        Returns:
        None
        """
        # Convert 'OCCUR_TIME' to datetime and extract the hour
        self.shooting_data['OCCUR_TIME'] = pd.to_datetime(self.shooting_data['OCCUR_TIME'], format='%H:%M:%S').dt.hour

        # Categorize time of day
        self.shooting_data['TIME_CATEGORY'] = pd.cut(
            self.shooting_data['OCCUR_TIME'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            right=False
        )

        # Visualize the distribution of shooting incidents by time category
        plt.figure(figsize=(10, 6))
        sns.countplot(x='TIME_CATEGORY', data=self.shooting_data, palette='Set2')
        plt.title('Distribution of Shooting Incidents by Time Category')
        plt.xlabel('Time Category')
        plt.ylabel('Count')
        plt.show()


class ShootingJurisdictionAnalysis:
    def __init__(self, data_url):
        """
        Initialize the ShootingJurisdictionAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def visualize_jurisdiction_distribution(self):
        """
        Visualize the distribution of shooting incidents by jurisdiction.

        Returns:
        None
        """
        # Count occurrences of each jurisdiction code
        jurisdiction_counts = self.shooting_data['JURISDICTION_CODE'].value_counts()

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x=jurisdiction_counts.index, y=jurisdiction_counts.values, palette='cubehelix')
        plt.title('Shooting Incidents by Jurisdiction')
        plt.xlabel('Jurisdiction Code')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=45)
        plt.show()


class TopPrecinctsAnalysis:
    def __init__(self, data_url):
        """
        Initialize the TopPrecinctsAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def visualize_top_precincts(self, top_n=20):
        """
        Visualize the top N precincts with the highest number of shooting incidents.

        Parameters:
        - top_n (int): Number of top precincts to display.

        Returns:
        None
        """
        # Count occurrences of each precinct
        precinct_counts = self.shooting_data['PRECINCT'].value_counts().head(top_n)

        # Plotting
        plt.figure(figsize=(12, 6))
        sns.barplot(x=precinct_counts.index, y=precinct_counts.values, palette='rocket')
        plt.title(f'Top {top_n} Precincts with Highest Number of Shooting Incidents')
        plt.xlabel('Precinct')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=45)
        plt.show()


class DayOfWeekAnalysis:
    def __init__(self, data_url):
        """
        Initialize the DayOfWeekAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)
        # Convert 'OCCUR_DATE' to datetime type
        self.shooting_data['OCCUR_DATE'] = pd.to_datetime(self.shooting_data['OCCUR_DATE'], errors='coerce')

    def visualize_shooting_distribution_by_day(self):
        """
        Visualize the distribution of shooting incidents by day of the week.

        Returns:
        None
        """
        # Extract the day of the week from the 'OCCUR_DATE' column
        self.shooting_data['DAY_OF_WEEK'] = self.shooting_data['OCCUR_DATE'].dt.day_name()

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.countplot(
            x='DAY_OF_WEEK',
            data=self.shooting_data,
            order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            palette='muted'
        )
        plt.title('Distribution of Shooting Incidents by Day of the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Incidents')
        plt.show()


class LocationBoroughAnalysis:
    def __init__(self, data_url):
        """
        Initialize the LocationBoroughAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def visualize_shooting_distribution_by_location_and_borough(self):
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
            data=self.shooting_data,
            palette='husl'
        )
        plt.title('Distribution of Shooting Incidents by Location and Borough')
        plt.xlabel('Location Description')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=90)
        plt.legend(title='Borough')
        plt.show()


class DayOfMonthAnalysis:
    def __init__(self, data_url):
        """
        Initialize the DayOfMonthAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)
        # Convert 'OCCUR_DATE' to datetime
        self.shooting_data['OCCUR_DATE'] = pd.to_datetime(self.shooting_data['OCCUR_DATE'], errors='coerce')

    def visualize_shooting_distribution_by_day_of_month(self):
        """
        Visualize the distribution of shooting incidents by day of the month.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.countplot(
            x=self.shooting_data['OCCUR_DATE'].dt.day,
            palette='muted'
        )
        plt.title('Distribution of Shooting Incidents by Day of the Month')
        plt.xlabel('Day of the Month')
        plt.ylabel('Number of Incidents')
        plt.show()


class BoroughJurisdictionAnalysis:
    def __init__(self, data_url):
        """
        Initialize the BoroughJurisdictionAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def visualize_shooting_distribution_by_borough_and_jurisdiction(self):
        """
        Visualize the distribution of shooting incidents by borough and jurisdiction code.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.countplot(x='BORO', hue='JURISDICTION_CODE', data=self.shooting_data, palette='deep')
        plt.title('Distribution of Shooting Incidents by Borough and Jurisdiction Code')
        plt.xlabel('Borough')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Jurisdiction Code')
        plt.show()

class VictimAgeSexAnalysis:
    def __init__(self, data_url):
        """
        Initialize the VictimAgeSexAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def visualize_shooting_distribution_by_age_and_sex(self):
        """
        Visualize the distribution of shooting incidents by victim age group and sex.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.countplot(x='VIC_AGE_GROUP', hue='VIC_SEX', data=self.shooting_data, palette='coolwarm')
        plt.title('Shooting Incidents by Victim Age Group and Sex')
        plt.xlabel('Victim Age Group')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Victim Sex')
        plt.xticks(rotation=90)
        plt.show()


class BoroughSuspectRaceAnalysis:
    def __init__(self, data_url):
        """
        Initialize the BoroughSuspectRaceAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def visualize_shooting_distribution_by_borough_and_race(self):
        """
        Visualize shooting incidents by borough and suspect race using a stacked bar plot.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.countplot(x='BORO', hue='PERP_RACE', data=self.shooting_data, palette='Set2')
        plt.title('Shooting Incidents by Borough and Suspect Race')
        plt.xlabel('Borough')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Suspect Race')
        plt.show()

class TimeSeriesAnalysis:
    def __init__(self, data_url):
        """
        Initialize the TimeSeriesAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)
        self.shooting_data['OCCUR_DATE'] = pd.to_datetime(self.shooting_data['OCCUR_DATE'])

    def visualize_monthly_trend_of_shooting_incidents(self):
        """
        Visualize the monthly trend of shooting incidents using a time series plot.

        Returns:
        None
        """
        # Resample data to get monthly counts
        monthly_counts = self.shooting_data.resample('M', on='OCCUR_DATE').size()

        # Plotting
        plt.figure(figsize=(14, 8))
        monthly_counts.plot(marker='o', linestyle='-', color='purple')
        plt.title('Monthly Trend of Shooting Incidents')
        plt.xlabel('Date')
        plt.ylabel('Number of Incidents')
        plt.show()

class BoroughLocationAnalysis:
    def __init__(self, data_url):
        """
        Initialize the BoroughLocationAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def visualize_shooting_distribution_by_borough_and_location(self):
        """
        Visualize shooting incidents by borough and location description using a count plot.

        Returns:
        None
        """
        plt.figure(figsize=(14, 8))
        sns.countplot(x='BORO', hue='LOCATION_DESC', data=self.shooting_data, palette='Pastel1')
        plt.title('Shooting Incidents by Borough and Location Description')
        plt.xlabel('Borough')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Location Description', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


class PrecinctJurisdictionAnalysis:
    def __init__(self, data_url):
        """
        Initialize the PrecinctJurisdictionAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def visualize_shooting_distribution_by_precinct_and_jurisdiction(self):
        """
        Visualize shooting incidents by precinct and jurisdiction code using a count plot.

        Returns:
        None
        """
        plt.figure(figsize=(14, 8))
        sns.countplot(x='PRECINCT', hue='JURISDICTION_CODE', data=self.shooting_data, palette='viridis')
        plt.title('Shooting Incidents by Precinct and Jurisdiction Code')
        plt.xlabel('Precinct')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Jurisdiction Code', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


class DayBoroughAnalysis:
    def __init__(self, data_url):
        """
        Initialize the DayBoroughAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def visualize_shooting_distribution_by_day_and_borough(self):
        """
        Visualize shooting incidents by day and borough using a stacked bar plot.

        Returns:
        None
        """
        # Convert 'OCCUR_DATE' to datetime
        self.shooting_data['OCCUR_DATE'] = pd.to_datetime(self.shooting_data['OCCUR_DATE'])

        # Extract the day of the week from the 'OCCUR_DATE' column
        self.shooting_data['DAY'] = self.shooting_data['OCCUR_DATE'].dt.day_name()

        # Group by day and borough and get counts
        day_borough_counts = self.shooting_data.groupby(['DAY', 'BORO']).size().unstack()

        # Plotting
        plt.figure(figsize=(14, 8))
        day_borough_counts.plot(kind='bar', stacked=True, cmap='tab10')
        plt.title('Shooting Incidents by Day and Borough')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Borough', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


class AgeGenderAnalysis:
    def __init__(self, data_url):
        """
        Initialize the AgeGenderAnalysis class.

        Parameters:
        - data_url (str): URL to the CSV file containing shooting incident data.
        """
        self.shooting_data = pd.read_csv(data_url)

    def visualize_shooting_distribution_by_age_and_gender(self):
        """
        Visualize shooting incidents by victim age group and gender using a count plot.

        Returns:
        None
        """
        # Plotting
        plt.figure(figsize=(14, 8))
        sns.countplot(x='VIC_AGE_GROUP', hue='VIC_SEX', data=self.shooting_data, palette='Set2')
        plt.title('Shooting Incidents by Victim Age Group and Gender')
        plt.xlabel('Victim Age Group')
        plt.ylabel('Number of Incidents')
        plt.legend(title='Victim Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


