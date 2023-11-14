package assignment22

// SQL packages
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField}


//ML packages
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.ClusteringEvaluator

// Loggers
import org.apache.log4j.Logger
import org.apache.log4j.Level

class Assignment {
  Logger.getLogger("org").setLevel(Level.OFF)
  // Spark Session
  val spark: SparkSession = SparkSession.builder()
    .appName("assignment")
    .config("spark.driver.host", "localhost")
    .master("local")
    .getOrCreate()

  // Schema dataD2
  val schemaD2 = new StructType(Array(
    StructField("a", DoubleType, nullable = true),
    StructField("b", DoubleType, nullable = true),
    StructField("LABEL", StringType, nullable = true)))


  // Read dataD2
  val dataD2: DataFrame = spark.read
    .option("delimiter", ",")
    .option("header", "true")
    .schema(schemaD2)
    .csv("data/dataD2.csv")

    println("Additional task 2: Efficient usage of data structures")
    dataD2.printSchema()
    dataD2.cache()
    dataD2.show(10)

  // Schema dataD2
  val schemaD3 = new StructType(Array(
    StructField("a", DoubleType, nullable = true),
    StructField("b", DoubleType, nullable = true),
    StructField("c", DoubleType, nullable = true),
    StructField("LABEL", StringType, nullable = true)))

  // Read dataD3
  val dataD3: DataFrame = spark.read
    .option("delimiter", ",")
    .option("header", value = true)
    .schema(schemaD3)
    .csv("data/dataD3.csv")

    dataD3.printSchema() //Additional task 2
    dataD3.cache()
    dataD3.show()

  val dataD2WithLabels: DataFrame = dataD2.
    withColumn("LABEL", when(col("LABEL") === "Fatal", 0.0)
    .otherwise(1.0))

  println("dataD2WithLabels.....")
  dataD2WithLabels.show()
  dataD2WithLabels.printSchema()

  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    println("Basic task 1: Basic 2D K-means")

    // Vector assembler for mapping column to features
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("features")

    println("Feature scaling [Additional task 4]")
    val featureScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("ScaledFeatures")

    // Pipeline to preprocess and learn the data
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, featureScaler))
    val pipelineFit = transformationPipeline.fit(df) // [Additional task 4]
    val transformedDf = pipelineFit.transform(df)

    // K-mean model
    val KMeans = new KMeans()
      .setK(k)
      .setFeaturesCol("ScaledFeatures")
      .setMaxIter(10)
      .setSeed(1L)


    // training the model
    val KModel: KMeansModel = KMeans.fit(transformedDf)

    //executing cluster results
    val clusters = KModel.clusterCenters
      .map(x => x.toArray)
      .map { case Array(f1, f2) => (f1, f2) }

    println(s"\n Total centroids = ${clusters.length} \n ")
    println("printing the clusters... ")
    clusters.foreach(println)
    clusters
  }


  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    println("Basic task 2: Three Dimensions")

    // Vector assembler for mapping column to features
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "c"))
      .setOutputCol("features")

    println("Additional task 4: ML (Machine Learning) pipeline")
    val featureScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("ScaledFeatures")

    // Pipeline to preprocess and learn the data
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler, featureScaler))
    val pipelineFit = transformationPipeline.fit(df) // [Additional task 4]
    val transformedDf = pipelineFit.transform(df)

    // K-mean model
    val KMeans = new KMeans()
      .setK(k)
      .setFeaturesCol("ScaledFeatures")
      .setMaxIter(10)
      .setSeed(1L)

    // training the model
    val KModel: KMeansModel = KMeans.fit(transformedDf)

    //executing cluster results
    val clusters = KModel.clusterCenters
      .map(x => x.toArray)
      .map { case Array(f1, f2, f3) => (f1, f2, f3) }

    println(s"\n Total centroids = ${clusters.length} \n ")
    println("printing the clusters... ")
    clusters.foreach(println)
    clusters
  }



  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    println("Basic task 3: Using Labels ")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "LABEL"))
      .setOutputCol("features")

    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler))
    val pipelineFit = transformationPipeline.fit(df)
    val transformedDf = pipelineFit.transform(df)

    val KMeans = new KMeans()
      .setK(k)
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setSeed(1L)

    val KModel: KMeansModel = KMeans.fit(transformedDf)

    val clustersTemp = KModel.clusterCenters
      .map(x => x.toArray)
      .map { case Array(f1, f2, f3) => (f1, f2, f3) }
      .sortBy(s => s._3)
      .take(2)

    val clusters = clustersTemp.map { case (f1, f2, f3) => (f1, f2) }
    clusters.foreach(println)
    clusters
  }


  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)] = {
    println("Basic task 4: Silhouette Method ")

    val vectorAssembler = new VectorAssembler()
          .setInputCols(Array("a", "b"))
          .setOutputCol("features")

    println("Additional task 4: ML (Machine Learning) pipeline")
    val featureScaler = new MinMaxScaler()
          .setInputCol("features")
          .setOutputCol("ScaledFeatures")

    val transformationPipeline = new Pipeline()
          .setStages(Array(vectorAssembler, featureScaler))
        val pipelineFit = transformationPipeline.fit(df) // [Additional task 4]
        val transformedDf = pipelineFit.transform(df)


    print("Additional task 1: Functional style")
    def calScore (k: Int, df: DataFrame) : (Int, Double) = {
          val evaluator = new ClusteringEvaluator()
          val kmeans = new KMeans()
                  .setFeaturesCol("ScaledFeatures")
                  .setK(k)
                  .setSeed(1L)
                  .setMaxIter(20)
          val pred = kmeans.fit(df).transform(df)
          (k, evaluator.evaluate(pred))
        }

    val score = (low to high).map(x => calScore(x, transformedDf)).toArray
    val x = List(score)

    print("Additional task 6: Scaling back to original scale")
    val originalScaleDf = spark.createDataFrame(score).withColumnRenamed("_1", "X_min")
                    .withColumnRenamed("_2", "X_max")
    originalScaleDf.show()
    score

  }

//  def dirtyData: Array[(Int, Double)] = {
//
//    val df: DataFrame = spark.read
//      .format("csv")
//      .option("header", "true")
//      .option("inferSchema", "true")
//      .load("data/dataD2_dirty.csv")
//
//    val assembledDf = new VectorAssembler()
//      .setInputCols(Array("a", "b"))
//      .setOutputCol("features")
//      .transform(df)
//
//    val scaledDf = new MinMaxScaler()
//      .setInputCol("features")
//      .setOutputCol("scaledFeatures")
//      .fit(assembledDf)
//      .transform(assembledDf)
//
//
//    }

}