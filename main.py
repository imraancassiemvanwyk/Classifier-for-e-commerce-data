try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, length
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
except ModuleNotFoundError as e:
    print("PySpark is not installed. Please install it by running 'pip install pyspark' and try again.")
    raise e

def create_spark_session(app_name="EcommerceClassifier") -> SparkSession:
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

def load_and_prepare_data(spark: SparkSession, file_path: str):
    """Load the dataset and clean the column names."""
    data = spark.read.csv(file_path, header=True, inferSchema=True)
    data = data.withColumnRenamed(data.columns[0], "Category")
    data = data.withColumnRenamed(data.columns[1], "Description")
    data = data.filter(col("Category").isNotNull() & col("Description").isNotNull())
    print("Data preview:")
    data.show(5)
    return data.repartition(4)  # Adjust partitioning for memory optimization

def preprocess_data(data):

    tokenizer = Tokenizer(inputCol="Description", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")
    label_indexer = StringIndexer(inputCol="Category", outputCol="label", handleInvalid="keep")
    return tokenizer, stopwords_remover, vectorizer, label_indexer

def build_pipeline(tokenizer: Tokenizer, stopwords_remover: StopWordsRemover, vectorizer: CountVectorizer, label_indexer: StringIndexer) -> Pipeline:
    classifier = LogisticRegression(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[
        tokenizer,
        stopwords_remover,
        vectorizer,
        label_indexer,
        classifier
    ])
    return pipeline

def evaluate_model(predictions) -> float:

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Model accuracy: {accuracy:.2f}")
    return accuracy

spark = create_spark_session()
data = load_and_prepare_data(spark, "ecommerceDataset.csv")
tokenizer, stopwords_remover, vectorizer, label_indexer = preprocess_data(data)
pipeline = build_pipeline(tokenizer, stopwords_remover, vectorizer, label_indexer)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)
predictions = model.transform(test_data)
evaluate_model(predictions)
spark.stop()
