package org.apache.spark.mllib.linalg.distributed
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Matrices, DenseMatrix, Matrix, DenseVector, Vector, SparseVector}
import org.apache.spark.mllib.linalg.distributed._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Axis, qr, svd, sum, SparseVector => BSV}
import math.{ceil, log}

import spray.json._
import DefaultJsonProtocol._
import java.io.{File, PrintWriter}

object CX {
  def fromBreeze(mat: BDM[Double]): DenseMatrix = {
    // FIXME: does not support strided matrices (e.g. views)
    new DenseMatrix(mat.rows, mat.cols, mat.data, mat.isTranspose)
  }

  // Returns `mat.transpose * mat * rhs`
  def multiplyGramianBy(mat: IndexedRowMatrix, rhs: DenseMatrix): DenseMatrix = {
    val rhsBrz = rhs.toBreeze.asInstanceOf[BDM[Double]]
    val result =
      mat.rows.treeAggregate(BDM.zeros[Double](mat.numCols.toInt, rhs.numCols))(
        seqOp = (U: BDM[Double], row: IndexedRow) => {
          val rowBrz = row.vector.toBreeze.asInstanceOf[BSV[Double]]
          val tmp: BDV[Double] = rhsBrz.t * rowBrz
          // performs a rank-1 update:
          //   U += outer(row.vector, tmp)
          for(ipos <- 0 until rowBrz.index.length) {
            val i = rowBrz.index(ipos)
            val ival = rowBrz.data(ipos)
            for(j <- 0 until tmp.length) {
              U(i, j) += ival * tmp(j)
            }
          }
          U
        },
        combOp = (U1, U2) => U1 += U2
      )
    fromBreeze(result)
  }

  def transposeMultiply(mat: IndexedRowMatrix, rhs: DenseMatrix): DenseMatrix = {
    require(mat.numRows == rhs.numRows)
    val rhsBrz = rhs.toBreeze.asInstanceOf[BDM[Double]]
    val result =
      mat.rows.treeAggregate(BDM.zeros[Double](mat.numCols.toInt, rhs.numCols))(
        seqOp = (U: BDM[Double], row: IndexedRow) => {
          val rowIdx = row.index.toInt
          val rowBrz = row.vector.toBreeze.asInstanceOf[BSV[Double]]
          // performs a rank-1 update:
          //   U += outer(row.vector, rhs(row.index, ::))
          for(ipos <- 0 until rowBrz.index.length) {
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

  // returns `mat.transpose * randn(m, rank)`
  def gaussianProjection(mat: IndexedRowMatrix, rank: Int): DenseMatrix = {
    val rng = new java.util.Random
    //val omega = DenseMatrix.randn(mat.numRows.toInt, rank, rng)
    //transposeMultiply(mat, DenseMatrix.randn(mat.numRows.toInt, rank, rng))
    DenseMatrix.randn(mat.numCols.toInt, rank, rng)
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("CX")
    conf.set("spark.driver.maxResultSize", "32g")
    conf.set("spark.eventLog.enabled", "true")
    conf.set("spark.eventLog.dir", "/root/spark/eventlogs")
    conf.set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)

    /* params */
    val numIters = 2
    // rank of approximation
    val rank = 8
    // extra slack to improve the approximation
    val slack = ceil(log(rank)/log(2)).toInt
    val k = rank + slack

    /* load matrix */
    val name = "Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked"
    val prefix = s"hdfs:///$name.mat"
    val shape = (131048, 8258911)

    /*
    val name = "Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked-100x100"
    val prefix = s"hdfs:///$name.mat"
    val shape = (9574, 3743324)
    */

    /*
    val name = "Lewis_Dalisay_Peltatum_20131115_PDX_Std_1"
    val prefix = "hdfs:///"
    val shape = (951, 781210)
    */

    /*
    val inpath = s"$prefix/$name.mat.csv"
    val nonzeros = sc.textFile(inpath).map(_.split(",")).
          map(x => new MatrixEntry(x(0).toLong, x(1).toLong, x(2).toDouble))
    val coomat = new CoordinateMatrix(nonzeros, shape._1, shape._2).transpose
    val mat = coomat.toIndexedRowMatrix()
    mat.rows.saveAsObjectFile(s"hdfs:///$name.rowmat")
    */

    val rows = sc.objectFile[IndexedRow](s"hdfs:///$name.rowmat")
    val mat = new IndexedRowMatrix(rows, shape._2, shape._1)
    mat.rows.cache()

    /* perform randomized SVD */
    var Y = gaussianProjection(mat, k).toBreeze.asInstanceOf[BDM[Double]]
    for(i <- 0 until numIters) {
      Y = multiplyGramianBy(mat, fromBreeze(Y)).toBreeze.asInstanceOf[BDM[Double]]
    }
    val Q = qr.reduced.justQ(Y)
    assert(Q.cols == k)
    val B = mat.multiply(fromBreeze(Q)).toBreeze.asInstanceOf[BDM[Double]].t
    val Bsvd = svd.reduced(B)
    // Since we computed randomized SVD of A', we unswap U and V here:
    val V = (Q * Bsvd.U).apply(::, 0 until rank)
    val S = Bsvd.S(0 until rank)
    val U = Bsvd.Vt(0 until rank, ::).t

    /* compute leverage scores */
    val rowlev = sum(U :^ 2.0, Axis._1)
    val rowp = rowlev / rank.toDouble
    val collev = sum(V :^ 2.0, Axis._1)
    val colp = collev / rank.toDouble
    assert(rowp.length == mat.numRows)
    assert(colp.length == mat.numCols)

    /* write output */
    val json = Map(
      "singvals" -> S.toArray.toSeq,
      "rowp" -> rowp.toArray.toSeq,
      "colp" -> colp.toArray.toSeq
    ).toJson
    val outw = new PrintWriter(new File("cx-out.json"))
    outw.println(json.compactPrint)
    outw.close()
  }
}
