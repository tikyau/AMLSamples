# General Predictive maintenance 

This sample is an implementation of the predictive maintenance real-world scenario template previously available in Azure Machine learning workbench using Azure Machine learning SDK.

- https://docs.microsoft.com/azure/machine-learning/preview/scenario-predictive-maintenance

- https://github.com/Azure/MachineLearningSamples-PredictiveMaintenance

## Changes introduced: 
  
  - Training and operationalization using Azure Machine Learning Service.
  - Model training on Azure Databricks cluster (gain of 1 order of magnitude in speed).
  - Prediction Serving using Azure container instance.
  - Minor Changes include: 
        
    *  Replacement of pandas dataframe manipulation to spark RDD.
    *  Addition of a features importance plot.
    *  Slight modification to existing EDA plots.

## Prerequisite:

Before running this notebook, make sure you have gone through the steps listed below:

- You have a workspace created https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-get-started 

- You have a development environment configured https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment

- You have an Azure Databricks cluster deployed with the following configuration:   
    * Databricks Runtime version: (latest stable release)(Scala 2.11)
    * Python version: 3
    * Driver/Worker type: Standard_DS13_v2
    * Python libraries installed:
        
         ```ipython==2.2.0, pyOpenSSL==16.0.0, psutil, azureml-sdk[databricks], cryptography==1.5```

