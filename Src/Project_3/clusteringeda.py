import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def loadAndExploreDataset():
    """
    Load the dataset and explore the distribution of crime incidents across different boroughs.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(
        "https://data.cityofnewyork.us/api/views/833y-fsy8/rows.csv?date=20231127&accessType=DOWNLOAD"
    )
    return df


def exploreBoroughDistribution(df):
    """
    Explore the distribution of crime incidents across different boroughs.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(
        x="BORO", data=df, order=df["BORO"].value_counts().index, palette="viridis"
    )
    plt.title("Distribution of Crime Incidents Across Boroughs")
    plt.xlabel("Borough")
    plt.ylabel("Number of Incidents")
    plt.show()


def exploreCrimeOverTime(df):
    """
    Explore crime incidents over time.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    df["OCCUR_DATE"] = pd.to_datetime(df["OCCUR_DATE"])
    df["YearMonth"] = df["OCCUR_DATE"].dt.to_period("M")
    plt.figure(figsize=(14, 6))
    sns.countplot(x="YearMonth", data=df.sort_values("OCCUR_DATE"), palette="viridis")
    plt.title("Trend of Crime Incidents Over Time")
    plt.xlabel("Year-Month")
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=90)
    plt.show()


def exploreLocationDescription(df):
    """
    Explore specific types of incidents based on location description.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(
        x="LOC_OF_OCCUR_DESC",
        data=df,
        order=df["LOC_OF_OCCUR_DESC"].value_counts().index,
        palette="viridis",
    )
    plt.title("Distribution of Crime Incidents by Location Description")
    plt.xlabel("Location Description")
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=90)
    plt.show()


def loadAndExploreDataset():
    """
    Load the dataset for exploring crime incident characteristics.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(
        "https://data.cityofnewyork.us/api/views/833y-fsy8/rows.csv?date=20231127&accessType=DOWNLOAD"
    )
    return df


def filterOutAgeGroups(df, ageGroupsToRemove):
    """
    Filter out rows with specified age groups.

    Args:
        df (pd.DataFrame): Input DataFrame.
        ageGroupsToRemove (list): List of age groups to be removed.

    Returns:
        pd.DataFrame: DataFrame after filtering out specified age groups.
    """
    dfFiltered = df.query(
        "PERP_AGE_GROUP not in @ageGroupsToRemove and VIC_AGE_GROUP not in @ageGroupsToRemove"
    )
    return dfFiltered


def explorePerpetratorAgeDistribution(df):
    """
    Explore the distribution of perpetrator ages.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df["PERP_AGE_GROUP"], bins="auto", kde=False, color="skyblue")
    plt.title("Distribution of Perpetrator Ages")
    plt.xlabel("Perpetrator Age Group")
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=45)
    plt.show()


def exploreVictimAgeDistribution(df):
    """
    Explore the distribution of victim ages.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df["VIC_AGE_GROUP"], bins="auto", kde=False, color="salmon")
    plt.title("Distribution of Victim Ages")
    plt.xlabel("Victim Age Group")
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=45)
    plt.show()


def explorePerpetratorRaceDistribution(df):
    """
    Explore the distribution of perpetrator races.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(
        x="PERP_RACE",
        data=df,
        order=df["PERP_RACE"].value_counts().index,
        palette="viridis",
    )
    plt.title("Distribution of Perpetrator Races")
    plt.xlabel("Perpetrator Race")
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=45)
    plt.show()


def exploreVictimRaceDistribution(df):
    """
    Explore the distribution of victim races.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(
        x="VIC_RACE",
        data=df,
        order=df["VIC_RACE"].value_counts().index,
        palette="viridis",
    )
    plt.title("Distribution of Victim Races")
    plt.xlabel("Victim Race")
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=45)
    plt.show()


def loadAndExploreDataset():
    """
    Load the dataset for exploring incident characteristics.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(
        "https://data.cityofnewyork.us/api/views/833y-fsy8/rows.csv?date=20231127&accessType=DOWNLOAD"
    )
    return df


def cleanAndPreprocessData(df):
    """
    Perform necessary data cleaning and preprocessing.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    df["OCCUR_DATE"] = pd.to_datetime(df["OCCUR_DATE"], errors="coerce")
    return df


def exploreIncidentDistributionOverTime(df):
    """
    Explore the distribution of incidents over time.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df["OCCUR_DATE"], kde=True, bins=30, color="skyblue")
    plt.title("Distribution of Incidents Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Incidents")
    plt.show()


def exploreIncidentDistributionAcrossBoroughs(df):
    """
    Explore the distribution of incidents across different boroughs.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(x="BORO", data=df, palette="viridis")
    plt.title("Distribution of Incidents Across Boroughs")
    plt.xlabel("Borough")
    plt.ylabel("Number of Incidents")
    plt.show()


def exploreRelationshipsBetweenNumericalFeatures(df):
    """
    Explore relationships between numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(12, 8))
    sns.pairplot(
        df[["Latitude", "Longitude", "OCCUR_DATE", "OCCUR_TIME", "VIC_AGE_GROUP"]],
        diag_kind="kde",
    )
    plt.suptitle("Pair Plot of Numerical Features", y=1.02)
    plt.show()
