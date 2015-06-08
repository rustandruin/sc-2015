name := "heromsi"

version := "0.0.1"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.1" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.3.1" % "provided"

libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"

libraryDependencies += "io.spray" %%  "spray-json" % "1.3.2"

lazy val submit = taskKey[Unit]("Submit CX job")
submit <<= (assembly in Compile) map {
  (jarFile: File) => s"src/runcx.sh ${jarFile}" !
}

lazy val runTest = taskKey[Unit]("Submit test job")
runTest <<= (assembly in Compile) map {
  (jarFile: File) => s"spark-submit --driver-memory 4G ${jarFile} test A_16K_1K.mtx B_1K_32.mtx" !
}
