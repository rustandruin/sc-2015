package org.apache.spark.mllib.linalg.distributed
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{Matrices, DenseMatrix, Matrix, DenseVector, Vector, SparseVector, Vectors}
import org.apache.spark.mllib.linalg.EigenValueDecomposition
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.sql.{SQLContext, Row => SQLRow}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Axis, qr, svd, sum, SparseVector => BSV}
import breeze.linalg.{norm, diag, accumulate}
import breeze.numerics.{sqrt => BrzSqrt}
import math.{ceil, log}
import scala.collection.mutable.ArrayBuffer

import spray.json._
import DefaultJsonProtocol._
import java.util.Arrays
import java.io.{DataOutputStream, BufferedOutputStream, FileOutputStream, File}

object SVDVariants {
  def fromBreeze(mat: BDM[Double]): DenseMatrix = {
    // FIXME: does not support strided matrices (e.g. views)
    new DenseMatrix(mat.rows, mat.cols, mat.data, mat.isTranspose)
  }

  def report(message: String, verbose: Boolean = false) = {
    if(verbose) {
      println("STATUS REPORT: " + message)
    }
  }

  def sampleCenteredColumns(mat : IndexedRowMatrix, numsamps: Int, probs: BDV[Double], rowavg: BDV[Double]) = {
    val rng = new java.util.Random
    val observations = List.fill(numsamps)(rng.nextDouble)
    val colCumProbs = accumulate(probs).toArray.zipWithIndex
    val keepIndices = observations.map( u => colCumProbs.find(_._1 >= u).getOrElse(Tuple2(1.0, mat.numCols.toInt))._2.asInstanceOf[Int] ).toArray

    def sampleFromRow(row : IndexedRow) = {
      val rowBrz = row.vector.toBreeze.asInstanceOf[BSV[Double]]
      val valBuffer = ArrayBuffer.empty[Double]
      for(indexpos <- keepIndices) {
        valBuffer += rowBrz(indexpos) - rowavg(indexpos)
      }
      new IndexedRow(row.index, new DenseVector(valBuffer.toArray))
    }
    // ugly hack to collect the matrix
    fromBreeze(new IndexedRowMatrix(mat.rows.map( row => sampleFromRow(row)), 
      mat.numRows, numsamps).toBreeze)
  }

  def multiplyCenteredMatBy(mat: IndexedRowMatrix, rhs: DenseMatrix, avg: BDV[Double]) : DenseMatrix = {
    def centeredRow( row : Vector) : Vector = {
      val temp = (row.toBreeze.asInstanceOf[BSV[Double]] - avg).toArray
      Vectors.dense(temp)
    }
    var rowmat = new IndexedRowMatrix( mat.rows.map(row => new IndexedRow(row.index, centeredRow(row.vector))),
      mat.numRows, mat.numCols.toInt)
    // hack to collapse distributed matrix to a local DenseMatrix
    fromBreeze(rowmat.multiply(rhs).toBreeze)
  }

  // Returns `(mat.transpose * mat - avg*avg.transpose) * rhs`
  // Note this IS NOT THE COVARIANCE, because the scaling is not correct
  def multiplyCovarianceBy(mat: IndexedRowMatrix, rhs: DenseMatrix, avg: BDV[Double]): DenseMatrix = {
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
    val tmp = rhsBrz.t * avg
    fromBreeze(result - avg * tmp.t)
  }

  // computes BA where B is a local matrix and A is distributed: let b_i denote the
  // ith col of B and a_i denote the ith row of A, then BA = sum(b_i a_i)
  def leftMultiplyCenteredMatrixBy(mat: IndexedRowMatrix, lhs: DenseMatrix, avg: BDV[Double]) : DenseMatrix = {
   val lhsBrz = lhs.toBreeze.asInstanceOf[BDM[Double]]
   val result = 
     mat.rows.treeAggregate(BDM.zeros[Double](lhs.numRows, mat.numCols.toInt))(
       seqOp = (U: BDM[Double], row: IndexedRow) => {
         val rowBrz = row.vector.toBreeze.asInstanceOf[BSV[Double]] - avg
         U += lhsBrz(::, row.index.toInt) * rowBrz.t
       },
       combOp = (U1, U2) => U1 += U2
     )
   fromBreeze(result)
  }

  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName("CX")
    conf.set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    appMain(sc, args)
  }

  def getRowMean(mat: IndexedRowMatrix) = {
    1.0/mat.numRows * mat.rows.treeAggregate(BDV.zeros[Double](mat.numCols.toInt))(
      seqOp = (avg: BDV[Double], row: IndexedRow) => {
        val rowBrz = row.vector.toBreeze.asInstanceOf[BSV[Double]]
        for (ipos <- 0 until rowBrz.index.length) {
          val idx = rowBrz.index(ipos)
          val rval = rowBrz.data(ipos)
          avg(idx) += rval
        }
        avg
      },
      combOp = (avg1, avg2) => avg1 += avg2
      )
  }

  def loadMSIData(sc: SparkContext, matkind: String, shape: Tuple2[Int, Int], inpath: String, nparts: Int = 0) = {

      val sqlctx = new org.apache.spark.sql.SQLContext(sc)
      import sqlctx.implicits._

      val mat0: IndexedRowMatrix = 
        if(matkind == "csv") {
          /* weird way to input data:
           * We assume the data is stored as an m-by-n matrix, with each
           * observation as a column
           * we pass in the matrix dimensions as (n,m) = shape
           * the following code loads the data into an n-by-m matrix
           * so that the observations are now rows of the matrix
           */
          val nonzeros = sc.textFile(inpath).map(_.split(",")).
          map(x => new MatrixEntry(x(1).toLong, x(0).toLong, x(2).toDouble))
          val coomat = new CoordinateMatrix(nonzeros, shape._1, shape._2)
          val mat = coomat.toIndexedRowMatrix()
          mat
        } else if(matkind == "idxrow") {
          val rows = sc.objectFile[IndexedRow](inpath)
          new IndexedRowMatrix(rows, shape._1, shape._2)
        } else if(matkind == "df") {
          val numRows = if(shape._1 != 0) shape._1 else sc.textFile(inpath + "/rowtab.txt").count.toInt
          val numCols = if(shape._2 != 0) shape._2 else sc.textFile(inpath + "/coltab.txt").count.toInt
          val rows =
            sqlctx.parquetFile(inpath + "/matrix.parquet").rdd.map {
              case SQLRow(index: Long, vector: Vector) =>
                new IndexedRow(index, vector)
            }
          new IndexedRowMatrix(rows, numRows, numCols)
        } else {
          throw new RuntimeException(s"unrecognized matkind: $matkind")
        }
        
      if (nparts == 0) {
        mat0
      } else {
        new IndexedRowMatrix(mat0.rows.coalesce(nparts), mat0.numRows, mat0.numCols.toInt) 
      }
  }

  /* get low-rank approximation to the centered input matrix using randomized SVD
   * algorithm. The rows of mat are the observations, 
   */
  def computeRSVD(mat: IndexedRowMatrix, rank: Int, slack: Int, numIters: Int,
    verbose: Boolean = false) : Tuple4[BDM[Double], BDV[Double], BDM[Double], BDV[Double]] = {
    val rng = new java.util.Random
    val k = rank + slack
    var Y = DenseMatrix.randn(mat.numCols.toInt, k, rng).toBreeze.asInstanceOf[BDM[Double]]
    val rowavg = getRowMean(mat)

    report("performing iterations", verbose)
    for(i <- 0 until numIters) {
      Y = multiplyCovarianceBy(mat, fromBreeze(Y), rowavg).toBreeze.asInstanceOf[BDM[Double]]
      Y = qr.reduced.justQ(Y)
    }
    report("done iterations", verbose)

    report("performing QR to find basis", verbose)
    val Q = qr.reduced.justQ(Y)
    report("done performing QR", verbose)
    assert(Q.cols == k)

    report("performing SVD on A*Q", verbose)
    var B = multiplyCenteredMatBy(mat, fromBreeze(Q), rowavg).toBreeze.asInstanceOf[BDM[Double]].t
    val Bsvd = svd.reduced(B)
    // Since we compute the randomized SVD of A', unswap U and V here to get back
    // svd(A) = U*S*V'
    val V = (Q * Bsvd.U).apply(::, 0 until rank)
    val S = Bsvd.S(0 until rank)
    val U = Bsvd.Vt(0 until rank, ::).t
    report("done performing SVD", verbose)

    // Don't care about ordering so skip the reordering necessary to get the
    // singular values in increasing order
    /*
    if (k > rank) {
      val lambda = S(k-1 until k-1-rank by -1)
      val Usorted = U(::, k-1 until k-1-rank by -1).copy
      val Vsorted = V(::, k-1 until k-1-rank by -1).copy
      (Usorted, lambda, Vsorted, rowavg)
    } else {
      val lambda = S((0 until k).reverse)
      val Usorted = U(::, (0 until k).reverse).copy
      val Vsorted = V(::, (0 until k).reverse).copy
      (Usorted, lambda, Vsorted, rowavg)
     }
     */
    (U, S, V, rowavg)
  }

  def appMain(sc: SparkContext, args: Array[String]) = {
    if(args.length < 8) {
      Console.err.println("Expected args: [csv|idxrow|df] inpath nrows ncols outpath rank slack niters [nparts]")
      System.exit(1)
    }

    val matkind = args(0)
    val inpath = args(1)
    val shape = (args(2).toInt, args(3).toInt)
    val outpath = args(4)

    // rank of approximation
    val rank = args(5).toInt

    // extra slack to improve the approximation
    val slack = args(6).toInt

    // number of power iterations to perform
    val numIters = args(7).toInt

    //input the data
    val nparts = if(args.length >= 9) { args(8).toInt } else {0}
    val mat: IndexedRowMatrix = loadMSIData(sc, matkind, shape, inpath, nparts)
    mat.rows.cache()

    // force materialization of the RDD so we can separate I/O from compute
    mat.rows.count()

    /* perform randomized SVD of centered A' */
    val rsvdResults = computeRSVD(mat, rank, slack, numIters, true)
    val rsvdU = rsvdResults._1
    val rsvdSingVals = rsvdResults._2
    val rsvdV = rsvdResults._3
    val mean = rsvdResults._4

    // compute the CX (on centered data, this uses the singvecs from PCA)
    val levProbs = sum(rsvdV :^ 2.0, Axis._1) / rank.toDouble
    report("Sampling the columns according to leverage scores", true)
    val colsMat = sampleCenteredColumns(mat, rank, levProbs, mean).toBreeze.asInstanceOf[BDM[Double]]
    report("Done sampling columns", true)
    report("Taking the QR of the columns", true)
    val qr.QR(q, r) = qr.reduced(colsMat)
    report ("Done with QR", true)
    val cxQ = q
    report("Getting X in the CX decomposition", true)
    val xMat = r \ leftMultiplyCenteredMatrixBy(mat, fromBreeze(cxQ.t), mean).toBreeze.asInstanceOf[BDM[Double]]
    report("Done forming X", true)

    // compute the truncated SVD by using PCA to get the right singular vectors
    // todo: port PROPACK here
    val tol = 1e-8
    val maxIter = 50
    val covOperator = (v : BDV[Double]) => multiplyCovarianceBy(mat, fromBreeze(v.toDenseMatrix).transpose, mean).toBreeze.asInstanceOf[BDM[Double]].toDenseVector
    report("Done with centered RSVD and centered CX, now computing truncated centered SVD", true)
    // find PCs first using EVD, then project data unto these and find left singular vectors
    report("Computing truncated EVD of covariance operator", true)
    val (lambda2, u2) = EigenValueDecomposition.symmetricEigs(covOperator, mat.numCols.toInt, rank, tol, maxIter)
    report("Done with truncated EVD of covariance operator", true)
    report("Extracting left singular vectors")
    val firstSVD = svd.reduced(multiplyCenteredMatBy(mat, fromBreeze(u2), mean).toBreeze.asInstanceOf[BDM[Double]])
    val secondSVD = svd.reduced(diag(firstSVD.S)*firstSVD.Vt*u2.t)
    val tsvdU = firstSVD.U * secondSVD.U
    val tsvdSingVals = secondSVD.S
    val tsvdV = secondSVD.Vt.t
    report("Done extracting left singular vectors")

    report("Computing the Frobenius norm relative errors of approximations of centered matrix", true)

    var frobNorm : Double = 0.0
    var rsvdFrobNormErr : Double = 0.0
    var cxFrobNormErr : Double = 0.0
    var tsvdFrobNormErr : Double = 0.0

    frobNorm = math.sqrt(mat.rows.map(row => math.pow(norm(row.vector.toBreeze.asInstanceOf[BSV[Double]] - mean), 2)).reduce( (x:Double, y: Double) => x + y))
    rsvdFrobNormErr = calcCenteredFrobNormErr(mat, rsvdU, diag(rsvdSingVals) * rsvdV.t ,mean)
    cxFrobNormErr = calcCenteredFrobNormErr(mat, colsMat, xMat, mean)
    tsvdFrobNormErr = calcCenteredFrobNormErr(mat, tsvdU, diag(tsvdSingVals)*tsvdV.t, mean)

    report(f"Frobenius norm of centered matrix: ${frobNorm}", true)
    report(f"RSVD relative Frobenius norm error: ${rsvdFrobNormErr/frobNorm}", true)
    report(f"CX relative Frobenius norm error: ${cxFrobNormErr/frobNorm}", true)
    report(f"TSVD relative Frobenius norm error: ${tsvdFrobNormErr/frobNorm}", true)

    val outf = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(new File(outpath))))
    report("Writing parameters and approximation errors to file", true)
    outf.writeInt(shape._1)
    outf.writeInt(shape._2)
    outf.writeInt(rank)
    outf.writeInt(slack)
    outf.writeInt(numIters)
    outf.writeInt(nparts)
    outf.writeDouble(frobNorm)
    outf.writeDouble(rsvdFrobNormErr)
    outf.writeDouble(cxFrobNormErr)
    outf.writeDouble(tsvdFrobNormErr)
    outf.close()
  }

  def calcCenteredFrobNormErr(mat: IndexedRowMatrix, lhsTall: BDM[Double], rhsFat: BDM[Double], mean: BDV[Double]) = {
    var accum : Double = 0
    math.sqrt(
      mat.rows.treeAggregate(accum)(
        seqOp = (partial: Double, row: IndexedRow) => {
          val reconstructed = (lhsTall(row.index.toInt, ::) * rhsFat).t
          partial + math.pow(norm(row.vector.toBreeze.asInstanceOf[BSV[Double]] - mean - reconstructed), 2)
        },
        combOp = (partial1, partial2) => partial1 + partial2
      )
    )
  }

  def dump(outf: DataOutputStream, v: BDV[Double]) = {
    outf.writeInt(v.length)
    for(i <- 0 until v.length) {
      outf.writeDouble(v(i))
    }
  }
  def loadMatrixA(sc: SparkContext, fn: String) = {
    val input = scala.io.Source.fromFile(fn).getLines()
    require(input.next() == "%%MatrixMarket matrix coordinate real general")
    val dims = input.next().split(' ').map(_.toInt)
    val seen = BDM.zeros[Int](dims(0), dims(1))
    val entries = input.map(line => {
      val toks = line.split(" ")
      val i = toks(0).toInt - 1
      val j = toks(1).toInt - 1
      val v = toks(2).toDouble
      require(toks.length == 3)
      new MatrixEntry(i, j, v)
    }).toSeq
    require(entries.length == dims(2))
    new CoordinateMatrix(sc.parallelize(entries, 1), dims(0), dims(1)).toIndexedRowMatrix
  }

  def loadMatrixB(fn: String) = {
    val input = scala.io.Source.fromFile(fn).getLines()
    require(input.next() == "%%MatrixMarket matrix coordinate real general")
    val dims = input.next().split(' ').map(_.toInt)
    val seen = BDM.zeros[Int](dims(0), dims(1))
    val result = BDM.zeros[Double](dims(0), dims(1))
    var count = 0
    input.foreach(line => {
      val toks = line.split(" ")
      require(toks.length == 3)
      val i = toks(0).toInt - 1
      val j = toks(1).toInt - 1
      val v = toks(2).toDouble
      require(i >= 0 && i < dims(0))
      require(j >= 0 && j < dims(1))
      require(seen(i, j) == 0)
      seen(i, j) = 1
      result(i, j) = v
      if(v != 0) count += 1
    })
    //assert(count == dims(2))
    fromBreeze(result)
  }

  def writeMatrix(mat: DenseMatrix, fn: String) = {
    val writer = new java.io.FileWriter(new java.io.File(fn))
    writer.write("%%MatrixMarket matrix coordinate real general\n")
    writer.write(s"${mat.numRows} ${mat.numCols} ${mat.numRows*mat.numCols}\n")
    for(i <- 0 until mat.numRows) {
      for(j <- 0 until mat.numCols) {
        writer.write(f"${i+1} ${j+1} ${mat(i, j)}%f\n")
      }
    }
    writer.close
  }
}
