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

  def transposeMultiply(mat: IndexedRowMatrix, rhs: DenseMatrix): DenseMatrix = {
    val rhsBrz = rhs.toBreeze.asInstanceOf[BDM[Double]]
    val result =
      mat.rows.treeAggregate(BDM.zeros[Double](mat.numCols.toInt, rhs.numCols))(
        seqOp = (U: BDM[Double], row: IndexedRow) => {
          val rowIdx = row.index.toInt
          val rowBrz = row.vector.toBreeze.asInstanceOf[BSV[Double]]
          // performs a rank-1 update:
          //   U += outer(row.vector, rhs(row.index, ::))
          for(ipos <- 0 until rowBrz.length) {
            val i = rowBrz.index(ipos)
            val ival = rowBrz.data(ipos)
            for(j <- 0 until rhs.numCols) {
              U(i, j) += ival * rhsBrz(rowIdx, j)
            }
          }
          U
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
    //val prefix = "hdfs:///"
    //val name = "Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked-100x100"
    val name = "Lewis_Dalisay_Peltatum_20131115_PDX_Std_1"
    val prefix = s"/home/jey/proj/openmsi/data/2014Nov15_PDX_IMS_imzML/$name"
    val inpath = s"$prefix/$name.mat.csv"
    val conf = new SparkConf().setAppName("CX")
    val sc = new SparkContext(conf)

    /* params */
    val numIters = 1
    val rank = 4      // rank of approximation
    val slack = rank  // extra slack to improve the approximation

    /* load matrix */
    val nonzeros = sc.textFile(inpath).map(_.split(",")).
          map(x => new MatrixEntry(x(0).toLong, x(1).toLong, x(2).toDouble))
    val coomat = new CoordinateMatrix(nonzeros, 951, 781210) // FIXME: magics
    val mat = coomat.toIndexedRowMatrix()
    mat.rows.cache()

    /* perform randomized SVD */
    var Y = gaussianProjection(mat, rank + slack).toBreeze
    for(i <- 0 until numIters) {
      Y = multiplyGramianBy(mat, fromBreeze(Y)).toBreeze.asInstanceOf[BDM[Double]]
    }
    assert(Y.cols == rank + slack)
    assert(Y.rows == 951)
    val Q = qr.justQ(Y)
    val B = transposeMultiply(mat, fromBreeze(Q)).transpose.toBreeze.asInstanceOf[BDM[Double]]
    val Bsvd = svd.reduced(B)
    val U = (Q * Bsvd.U).apply(::, 0 until rank)
    val S = Bsvd.S(0 until rank)
    val Vt = Bsvd.Vt(0 until rank, ::)

    /* compute leverage scores */
    val rowlev = sum(U :^ 2.0, Axis._1)
    val rowp = rowlev / rank.toDouble
    val collev = sum(Vt :^ 2.0, Axis._1)
    val colp = collev / rank.toDouble

    println("S:\n")
    println(S)
    println("RowP:")
    println(rowp)
    println("\nColP:")
    println(colp)
  }
}
