# Databricks notebook source
# MAGIC %md
# MAGIC # European Soccer Events Analysis - Machine Learning
# MAGIC 
# MAGIC As we saw, doing descriptive analysis on big data (like above) has been made super easy with Spark SQL and Databricks. But what if you’re a data scientist who’s looking at the same data to find combinations of on-field playing conditions that lead to “goals”?
# MAGIC 
# MAGIC We’ll now create a third notebook from that perspective, and see how one could fit a [Gradient-boosted tree](https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#gradient-boosted-tree-classifier) classifier Spark ML model on the game events training dataset. In this case, our binary classification label will be field “is_goal”, and we’ll use a mix of categorical features like “event_type_str”, “event_team”, “shot_place_str”, “location_str”, “assist_method_str”, “situation_str” and “country_code”.
# MAGIC 
# MAGIC First, we need to do the necessary imports from Spark ML:

# COMMAND ----------

# MAGIC %sql
# MAGIC USE SoccerDemo;

# COMMAND ----------

# DBTITLE 1,Explore data to establish features
# MAGIC %sql
# MAGIC SELECT * FROM GAME_EVENTS;

# COMMAND ----------

# DBTITLE 1, Create dataset for model training and prediction
gameEventsDf = spark.sql("select event_type_str, event_team, shot_place_str, location_str, assist_method_str, situation_str, country_code, is_goal from game_events")

display(gameEventsDf)

# COMMAND ----------

# DBTITLE 1,Necessary imports for Spark ML pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# DBTITLE 1, Convert our categorical feature columns to a single binary vector
# MAGIC %md
# MAGIC Then the following four-step process is required to convert our categorical feature columns to a single binary vector:
# MAGIC 
# MAGIC - Convert string features to indices using StringIndexer
# MAGIC - Transform feature indices to binary vectors using OneHotEncoder
# MAGIC - Assemble different binary vector columns into a single vector using VectorAssembler

# COMMAND ----------

# DBTITLE 1,Create a list of categorical features
categFeatures = ["event_type_str", "event_team", "shot_place_str", "location_str", "assist_method_str", "situation_str", "country_code"]

# COMMAND ----------

# DBTITLE 1,Encode categorical string cols to label indices
stringIndexers = [StringIndexer().setInputCol(baseFeature).setOutputCol(baseFeature + "_idx") for baseFeature in categFeatures]

# COMMAND ----------

# DBTITLE 1, Convert categorical label indices to binary vectors
encoders = [OneHotEncoder().setInputCol(baseFeature + "_idx").setOutputCol(baseFeature + "_vec") for baseFeature in categFeatures]

# COMMAND ----------

# DBTITLE 1,Combine all columns into a single feature vector
featureAssembler = VectorAssembler()
featureAssembler.setInputCols([baseFeature + "_vec" for baseFeature in categFeatures])
featureAssembler.setOutputCol("features")

# COMMAND ----------

# DBTITLE 1,Create Spark ML pipeline using a GBT classifier
# MAGIC %md
# MAGIC Finally, we’ll create a Spark ML Pipeline using the above transformers and the GBT classifier. We’ll divide the game events data into training and test datasets, and fit the pipeline to the former.

# COMMAND ----------

gbtClassifier = GBTClassifier(labelCol="is_goal", featuresCol="features", maxDepth=5, maxIter=20)

pipelineStages = stringIndexers + encoders + [featureAssembler, gbtClassifier]
pipeline = Pipeline(stages=pipelineStages)

# COMMAND ----------

# DBTITLE 1,Split dataset into training/test, and create a model from training data
(trainingData, testData) = gameEventsDf.randomSplit([0.75, 0.25])
model = pipeline.fit(trainingData)

# COMMAND ----------

# DBTITLE 1,Validate the model on test data, display predictions
# MAGIC %md
# MAGIC Now we can validate our classification model by running inference on the test dataset. We could compare the predicted label with actual label one by one, but that could be a painful process for lots of test data. For scalable model evaluation in this case, we can use BinaryClassificationEvaluator with area under ROC metric. One could also use the Precision-Recall Curve as an evaluation metric. If it was a multi-classification problem, we could’ve used the MulticlassClassificationEvaluator.

# COMMAND ----------

predictions = model.transform(testData)
display(predictions.select("prediction", "is_goal", "features"))

# COMMAND ----------

# DBTITLE 1,Evaluate the model using areaUnderROC metric
evaluator = BinaryClassificationEvaluator(
    labelCol="is_goal", rawPredictionCol="prediction")
evaluator.evaluate(predictions)