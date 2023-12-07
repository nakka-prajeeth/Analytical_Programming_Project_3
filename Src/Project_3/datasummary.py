class DatasetInfo:
    def __init__(self, dataUrl):
        """
        A class to load a dataset and provide information about it.

        Parameters:
        - dataUrl (str): The URL to the dataset.

        Attributes:
        - data (pd.DataFrame): The loaded dataset.
        """

        self.data = pd.read_csv(dataUrl)

    def displayTotalInstances(self):
        """
        Display the total number of instances in the loaded dataset.

        Returns:
        None
        """
        totalInstances = len(self.data)
        print(f"Total Instances in the Dataset: {totalInstances}")
