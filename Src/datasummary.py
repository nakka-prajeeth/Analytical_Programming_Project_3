import pandas as pd

class DatasetInfo:
    def __init__(self, data_url):
        """
        A class to load a dataset and provide information about it.

        Parameters:
        - data_url (str): The URL to the dataset.

        Attributes:
        - data (pd.DataFrame): The loaded dataset.
        """

        self.data = pd.read_csv(data_url)

    def display_total_instances(self):
        """
        Display the total number of instances in the loaded dataset.

        Returns:
        None
        """
        total_instances = len(self.data)
        print(f'Total Instances in the Dataset: {total_instances}')
