# Databricks notebook source
# MAGIC %md
# MAGIC # European Soccer Events Analysis - Summary
# MAGIC 
# MAGIC We demonstrated how to build the three functional components of data engineering, data analysis, and machine learning using the Databricks Unified Analytics Platform.  We’ve illustrated how you can run your ETL, analysis, and visualization, and machine learning pipelines all within a single Databricks notebook. By removing the data engineering complexities commonly associated with such data pipelines with the Databricks Unified Analytics Platform, this allows different sets of users i.e. data engineers, data analysts, and data scientists to easily work together to find hidden value in big data from any sports.
# MAGIC 
# MAGIC But this is just the first step. A sports or media organization could do more by running model-based inference on real-time streaming data processed using Structured Streaming, in order to provide targeted content to its users. And then there are many other ways to combine different Spark/Databricks technologies, to solve different big data problems in sport and media industries.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup

# COMMAND ----------

# DBTITLE 1,Cleanup databases
# MAGIC %sql 
# MAGIC DROP DATABASE IF EXISTS SoccerDemo CASCADE;

# COMMAND ----------

# DBTITLE 1,Cleanup Data Lake
dbutils.fs.rm("abfss://europesoccer@demomdwhdls01.dfs.core.windows.net/curated/game_events.csv", True)
dbutils.fs.rm("abfss://europesoccer@demomdwhdls01.dfs.core.windows.net/curated/game_events.parquet", True)