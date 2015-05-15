name := "Spark MSI"

version := "0.0.1"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.1"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.3.1"

lazy val submit = taskKey[Unit]("Submits Spark job")
submit <<= (Keys.`package` in Compile) map {
  (jarFile: File) => s"spark-submit --verbose --driver-memory 64G ${jarFile}" !
}
