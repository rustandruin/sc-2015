package org.apache.spark.mllib.linalg.distributed
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Matrices, DenseMatrix, Matrix, DenseVector, Vector}
import org.apache.spark.mllib.linalg.distributed._
import breeze.linalg.{Matrix => BM, DenseMatrix => BDM, DenseVector => BDV, Axis, qr, svd, sum, axpy, SparseVector => BSV}

object CX {
  def fromBreeze(mat: BDM[Double]): DenseMatrix = {
    new DenseMatrix(mat.rows, mat.cols, mat.data, mat.isTranspose)
  }

  def multiplyGramianBy(mat: IndexedRowMatrix, rhs: DenseMatrix): DenseMatrix = {
    val rhsBrz = rhs.toBreeze.asInstanceOf[BDM[Double]]
    val result =
      mat.rows.treeAggregate(BDM.zeros[Double](mat.numRows.toInt, rhs.numCols))(
        seqOp = (U: BDM[Double], row: IndexedRow) => {
          val rowBrz = row.vector.toBreeze.asInstanceOf[BSV[Double]]
          U += rowBrz.t * rowBrz * rhsBrz
        },
        combOp = (U1, U2) => U1 += U2
      )
    fromBreeze(result)
  }

  def gaussianProjection(mat: IndexedRowMatrix, rank: Int): IndexedRowMatrix = {
    val rng = new java.util.Random
    mat.multiply(DenseMatrix.randn(mat.numCols.toInt, rank, rng))
  }

  def main(args: Array[String]) {
    val prefix = "hdfs:///"
    val name = "Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked-100x100"
    val inpath = s"$prefix/$name.mat.csv"
    val conf = new SparkConf().setAppName("CX")
    conf.setMaster("local[4]").set("spark.driver.memory", "8G")
    val sc = new SparkContext(conf)

    /* params */
    val numIters = 2
    val rank = 8  // rank of approximation
    val slack = 10  // extra slack to improve the approximation
    val reo = 4     // reorthogonalize after this many iters

    /* load matrix */
    val nonzeros = sc.textFile(inpath).map(_.split(",")).
          map(x => new MatrixEntry(x(1).toLong, x(0).toLong, x(2).toDouble))
    val coomat = new CoordinateMatrix(nonzeros, 3743324, 9574) // FIXME: magics
    val mat = coomat.toIndexedRowMatrix()
    mat.rows.cache()

    /* approximate principal subspace */
    var B = gaussianProjection(mat, rank + slack).toBreeze
    for(i <- 0 to numIters) {
      if(i % reo == reo-1) {
        println("reorth")
        B = qr.justQ(B)
      }
      B = multiplyGramianBy(mat, fromBreeze(B)).toBreeze.asInstanceOf[BDM[Double]]
    }

    /* compute leverage scores */
    val U = svd.reduced(B).U(::, 0 until rank)
    // lev = np.sum(U[:,:k]**2,axis=1)
    val lev = sum(U :^ 2.0, Axis._0)
    val p = lev / rank.toDouble

    println(p)
  }
}
