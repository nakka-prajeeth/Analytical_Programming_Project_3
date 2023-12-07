import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def loadShootingData(url):
    """
    Load the shooting incident dataset from the provided URL.

    Parameters:
    - url (str): The URL to the dataset.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(url)


def selectSpatialFeatures(df, spatialFeatures):
    """
    Select relevant spatial features from the dataset.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - spatialFeatures (list): List of column names representing spatial features.

    Returns:
    - pd.DataFrame: Subset of the dataframe containing selected spatial features.
    """
    # Ensure that the column names are correctly matched, including any leading/trailing whitespaces
    return df[spatialFeatures].copy()


def performKMeansClustering(xSpatial, nClusters=5, nInit=10):
    """
    Perform K-Means clustering on spatial features.

    Parameters:
    - xSpatial (pd.DataFrame): Spatial features dataframe.
    - nClusters (int): Number of clusters for K-Means.
    - nInit (int): Number of times K-Means algorithm will be run with different centroid seeds.

    Returns:
    - pd.Series: Cluster labels assigned by K-Means.
    """
    kmeansSpatial = KMeans(n_clusters=nClusters, n_init=nInit)
    return kmeansSpatial.fit_predict(xSpatial)


def visualizeClusters(xSpatial, clusterLabels):
    """
    Visualize clusters on a map.

    Parameters:
    - xSpatial (pd.DataFrame): Spatial features dataframe.
    - clusterLabels (pd.Series): Cluster labels assigned by K-Means.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        xSpatial["Longitude"],
        xSpatial["Latitude"],
        c=clusterLabels,
        cmap="viridis",
        alpha=0.5,
    )
    plt.title("K-Means Clustering of Shooting Incidents in New York")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def loadDataset(url):
    """
    Load the dataset from the given URL.

    Parameters:
    - url (str): URL of the dataset.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(url)
    return df


def displayColumns(df):
    """
    Display the columns of the dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    """
    print(df.columns)


def selectTemporalFeatures(df, temporalFeatures):
    """
    Select relevant temporal features from the dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - temporalFeatures (list): List of column names for temporal features.

    Returns:
    - pd.DataFrame: Dataframe containing selected temporal features.
    """
    X_temporal = df[temporalFeatures].copy()
    X_temporal["OCCUR_DATE"] = pd.to_datetime(X_temporal["OCCUR_DATE"])
    X_temporal["OCCUR_TIME"] = pd.to_datetime(
        X_temporal["OCCUR_TIME"], format="%H:%M:%S"
    ).dt.time
    X_temporal["datetime"] = pd.to_datetime(
        X_temporal["OCCUR_DATE"].astype(str)
        + " "
        + X_temporal["OCCUR_TIME"].astype(str)
    )
    X_temporal = X_temporal[["datetime"]]
    X_temporal = X_temporal.dropna()
    return X_temporal


def performKMeansClustering(X_temporal, nClusters=5, nInit=10):
    """
    Perform K-Means clustering on temporal features.

    Parameters:
    - X_temporal (pd.DataFrame): Temporal features dataframe.
    - nClusters (int): Number of clusters for K-Means.
    - nInit (int): Number of times K-Means will be run with different centroid seeds.

    Returns:
    - pd.Series: Cluster labels assigned by K-Means.
    """
    kmeansTemporal = KMeans(n_clusters=nClusters, n_init=nInit)
    return kmeansTemporal.fit_predict(X_temporal[["datetime"]])


def visualizeClustersOverTime(X_temporal, clusterLabels):
    """
    Visualize the clusters over time.

    Parameters:
    - X_temporal (pd.DataFrame): Temporal features dataframe.
    - clusterLabels (pd.Series): Cluster labels assigned by K-Means.
    """
    if "cluster" not in X_temporal.columns:
        X_temporal["cluster"] = clusterLabels

    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_temporal.index,
        X_temporal["cluster"],
        c=X_temporal["cluster"],
        cmap="viridis",
        alpha=0.5,
    )
    plt.title("K-Means Clustering of Shooting Incidents Over Time")
    plt.xlabel("Data Point Index")
    plt.ylabel("Cluster")
    plt.show()


def loadDataset(url):
    """
    Loading the dataset from the given URL.

    Parameters:
    - url (str): URL of the dataset.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(url)
    return df


def displayColumns(df):
    """
    Displaying the columns of the dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    """
    print(df.columns)


def selectAdditionalFeatures(df, features):
    """
    Selecting relevant features for clustering.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - features (list): List of column names for additional features.

    Returns:
    - pd.DataFrame: Dataframe containing selected additional features.
    """
    xAdditional = df[features].copy()
    xAdditional = pd.get_dummies(xAdditional, columns=features, drop_first=True)
    xAdditional = xAdditional.dropna()
    return xAdditional


def performKMeansClustering(xAdditional, nClusters=5, nInit=10):
    """
    Performing K-Means clustering on additional features.

    Parameters:
    - xAdditional (pd.DataFrame): Additional features dataframe.
    - nClusters (int): Number of clusters for K-Means.
    - nInit (int): Number of times K-Means will be run with different centroid seeds.

    Returns:
    - pd.Series: Cluster labels assigned by K-Means.
    """
    kmeansAdditional = KMeans(n_clusters=nClusters, n_init=nInit)
    return kmeansAdditional.fit_predict(
        xAdditional.drop(columns="cluster", errors="ignore")
    )


def visualizeClustersBasedOnAdditionalFeatures(xAdditional):
    """
    Visualizing the clusters based on additional features.

    Parameters:
    - xAdditional (pd.DataFrame): Additional features dataframe.
    """
    fig, axes = plt.subplots(
        len(xAdditional.columns) - 1,
        1,
        figsize=(10, 6 * (len(xAdditional.columns) - 1)),
    )

    for i, feature in enumerate(xAdditional.columns[:-1]):
        axes[i].scatter(
            range(len(xAdditional)),
            xAdditional[feature],
            c=xAdditional["cluster"],
            cmap="viridis",
            alpha=0.5,
        )
        axes[i].set_title(f"K-Means Clustering based on {feature}")
        axes[i].set_xlabel("Data Point Index")
        axes[i].set_ylabel(feature)

    plt.tight_layout()
    plt.show()


def loadAndPreprocessData():
    """
    Load the dataset from the given URL and preprocess it for clustering.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with relevant features.
    """
    url = "https://data.cityofnewyork.us/api/views/833y-fsy8/rows.csv?date=20231127&accessType=DOWNLOAD"
    df = pd.read_csv(url)
    return df


def selectFeatures(df):
    """
    Select relevant features for clustering based on research question and dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing selected features.
    """
    temporalFeatures = ["OCCUR_DATE", "OCCUR_TIME"]
    additionalFeatures = ["VIC_AGE_GROUP", "PERP_AGE_GROUP", "BORO"]
    xCombined = df[temporalFeatures + additionalFeatures].copy()
    xCombined["OCCUR_TIME"] = (
        pd.to_datetime(xCombined["OCCUR_TIME"], format="%H:%M:%S").dt.hour * 60
        + pd.to_datetime(xCombined["OCCUR_TIME"], format="%H:%M:%S").dt.minute
    )
    xCombined = xCombined.drop(columns=["OCCUR_DATE"])
    xCombined = pd.get_dummies(
        xCombined, columns=["VIC_AGE_GROUP", "PERP_AGE_GROUP", "BORO"], drop_first=True
    )
    columnsToRemove = [
        "PERP_AGE_GROUP_1020",
        "PERP_AGE_GROUP_224",
        "PERP_AGE_GROUP_940",
    ]
    xCombined = xCombined.drop(columns=columnsToRemove, errors="ignore")
    xCombined = xCombined.dropna()
    return xCombined


def performKMeansClustering(xCombined, nClusters=5, nInit=10):
    """
    Perform K-Means clustering on the given DataFrame.

    Args:
        xCombined (pd.DataFrame): DataFrame with features for clustering.
        nClusters (int): Number of clusters to form.
        nInit (int): Number of times K-Means algorithm will be run with different centroid seeds.

    Returns:
        pd.DataFrame: DataFrame with an additional 'cluster' column indicating the cluster assignment.
    """
    kMeansCombined = KMeans(n_clusters=nClusters, n_init=nInit)
    xCombined["cluster"] = kMeansCombined.fit_predict(xCombined)
    return xCombined


def visualizeClusters(xCombined):
    """
    Visualize clusters based on combined features using scatter plots.

    Args:
        xCombined (pd.DataFrame): DataFrame with features and cluster assignments.
    """
    fig, axes = plt.subplots(
        len(xCombined.columns) - 1, 1, figsize=(10, 6 * (len(xCombined.columns) - 1))
    )

    for i, feature in enumerate(xCombined.columns[:-1]):
        axes[i].scatter(
            range(len(xCombined)),
            xCombined[feature],
            c=xCombined["cluster"],
            cmap="viridis",
            alpha=0.5,
        )
        axes[i].set_title(f"K-Means Clustering based on {feature}")
        axes[i].set_xlabel("Data Point Index")
        axes[i].set_ylabel(feature)

    plt.tight_layout()
    plt.show()


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def loadAndPreprocessData(url):
    """
    Load and preprocess the dataset for clustering.

    Args:
        url (str): URL of the dataset.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(url)
    return df


def displayColumns(df):
    """
    Display the columns of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    print(df.columns)


def selectAndInspectFeatures(df, additionalFeatures):
    """
    Select relevant features for clustering and inspect unique values before and after one-hot encoding.

    Args:
        df (pd.DataFrame): Input DataFrame.
        additionalFeatures (list): List of additional features.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded features.
    """
    print("Unique values before one-hot encoding:")
    for feature in additionalFeatures:
        uniqueValuesBeforeEncoding = df[feature].unique()
        print(f"{feature}: {uniqueValuesBeforeEncoding}")

    dfOneHot = pd.get_dummies(
        df[additionalFeatures], columns=additionalFeatures, drop_first=True
    )

    print("\nUnique values after one-hot encoding:")
    for column in dfOneHot.columns:
        uniqueValuesAfterEncoding = dfOneHot[column].unique()
        print(f"{column}: {uniqueValuesAfterEncoding}")

    return dfOneHot


def performKMeansClustering(dfOneHot, nClusters=5, nInit=10):
    """
    Perform K-Means clustering on the given DataFrame.

    Args:
        dfOneHot (pd.DataFrame): DataFrame with one-hot encoded features.
        nClusters (int): Number of clusters to form.
        nInit (int): Number of times K-Means algorithm will be run with different centroid seeds.

    Returns:
        pd.DataFrame: DataFrame with an additional 'cluster' column indicating the cluster assignment.
    """
    kMeansAdditional = KMeans(n_clusters=nClusters, n_init=nInit)
    dfOneHot["cluster"] = kMeansAdditional.fit_predict(dfOneHot)
    return dfOneHot


def visualizeClusters(dfOneHot):
    """
    Visualize clusters based on additional features using scatter plots.

    Args:
        dfOneHot (pd.DataFrame): DataFrame with one-hot encoded features and cluster assignments.
    """
    fig, axes = plt.subplots(
        len(dfOneHot.columns) - 1, 1, figsize=(10, 6 * (len(dfOneHot.columns) - 1))
    )

    for i, feature in enumerate(dfOneHot.columns[:-1]):
        axes[i].scatter(
            range(len(dfOneHot)),
            dfOneHot[feature],
            c=dfOneHot["cluster"],
            cmap="viridis",
            alpha=0.5,
        )
        axes[i].set_title(f"K-Means Clustering based on {feature}")
        axes[i].set_xlabel("Data Point Index")
        axes[i].set_ylabel(feature)

    plt.tight_layout()
    plt.show()


def loadAndPreprocessData(url):
    """
    Load and preprocess the dataset for clustering.

    Args:
        url (str): URL of the dataset.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(url)
    return df


def displayColumns(df):
    """
    Display the columns of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    print(df.columns)


def selectAndCombineFeatures(df, temporalFeatures, additionalFeatures):
    """
    Select and combine relevant features for clustering.

    Args:
        df (pd.DataFrame): Input DataFrame.
        temporalFeatures (list): List of temporal features.
        additionalFeatures (list): List of additional features.

    Returns:
        pd.DataFrame: Combined DataFrame with selected features.
    """
    dfCombined = df[temporalFeatures + additionalFeatures].copy()
    dfCombined["OCCUR_TIME"] = (
        pd.to_datetime(dfCombined["OCCUR_TIME"], format="%H:%M:%S").dt.hour * 60
        + pd.to_datetime(dfCombined["OCCUR_TIME"], format="%H:%M:%S").dt.minute
    )
    dfCombined = dfCombined.drop(columns=["OCCUR_DATE"])
    return dfCombined


def oneHotEncodeAndRemoveColumns(dfCombined, columnsToRemove):
    """
    One-hot encode categorical features and remove specific one-hot encoded columns.

    Args:
        dfCombined (pd.DataFrame): Combined DataFrame with selected features.
        columnsToRemove (list): List of columns to be removed.

    Returns:
        pd.DataFrame: DataFrame after one-hot encoding and column removal.
    """
    dfOneHot = pd.get_dummies(
        dfCombined, columns=["VIC_AGE_GROUP", "PERP_AGE_GROUP", "BORO"], drop_first=True
    )
    dfOneHot = dfOneHot.drop(columns=columnsToRemove, errors="ignore")
    return dfOneHot


def inspectUniqueValues(dfCombined, dfOneHot, additionalFeatures):
    """
    Inspect unique values before and after one-hot encoding.

    Args:
        dfCombined (pd.DataFrame): Combined DataFrame with selected features.
        dfOneHot (pd.DataFrame): DataFrame after one-hot encoding and column removal.
        additionalFeatures (list): List of additional features.
    """
    print("\nUnique values before one-hot encoding:")
    for feature in additionalFeatures:
        uniqueValuesBeforeEncoding = dfCombined[feature].unique()
        print(f"{feature}: {uniqueValuesBeforeEncoding}")

    print("\nUnique values after one-hot encoding:")
    for column in dfOneHot.columns:
        uniqueValuesAfterEncoding = dfOneHot[column].unique()
        print(f"{column}: {uniqueValuesAfterEncoding}")


def performKMeansClustering(dfOneHot, nClusters=5, nInit=10):
    """
    Perform K-Means clustering on the given DataFrame.

    Args:
        dfOneHot (pd.DataFrame): DataFrame after one-hot encoding and column removal.
        nClusters (int): Number of clusters to form.
        nInit (int): Number of times K-Means algorithm will be run with different centroid seeds.

    Returns:
        pd.DataFrame: DataFrame with an additional 'cluster' column indicating the cluster assignment.
    """
    kMeansCombined = KMeans(n_clusters=nClusters, n_init=nInit)
    dfOneHot["cluster"] = kMeansCombined.fit_predict(dfOneHot)
    return dfOneHot


def visualizeClusters(dfOneHot):
    """
    Visualize clusters based on combined features using scatter plots.

    Args:
        dfOneHot (pd.DataFrame): DataFrame with one-hot encoded features and cluster assignments.
    """
    fig, axes = plt.subplots(
        len(dfOneHot.columns) - 1, 1, figsize=(10, 6 * (len(dfOneHot.columns) - 1))
    )

    for i, feature in enumerate(dfOneHot.columns[:-1]):
        axes[i].scatter(
            range(len(dfOneHot)),
            dfOneHot[feature],
            c=dfOneHot["cluster"],
            cmap="viridis",
            alpha=0.5,
        )
        axes[i].set_title(f"K-Means Clustering based on {feature}")
        axes[i].set_xlabel("Data Point Index")
        axes[i].set_ylabel(feature)

    plt.tight_layout()
    plt.show()


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
