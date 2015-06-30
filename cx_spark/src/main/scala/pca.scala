package org.apache.spark.mllib.linalg.distributed

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Matrices, DenseMatrix, Matrix, DenseVector, Vector, SparseVector}
import org.apache.spark.mllib.linalg.EigenValueDecomposition
import org.apache.spark.mllib.linalg.distributed._

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Axis, qr, svd, sum, SparseVector => BSV}
import breeze.linalg.{norm, eigSym, diag, accumulate}

//import edu.berkeley.cs.amplab.mlmatrix.{RowPartitionedMatrix, TSQR}

import math.{ceil, log, sqrt}

import spray.json._
import DefaultJsonProtocol._
import java.util.Arrays
import java.io.{DataOutputStream, BufferedOutputStream, FileOutputStream, File}
import org.apache.spark.sql.{SQLContext, Row => SQLRow}

//import grizzled.slf4j.Logger

object PCAvariants { 

  def fromBreeze(mat: BDM[Double]): DenseMatrix = {
    // FIXME: does not support strided matrices (e.g. views)
    new DenseMatrix(mat.rows, mat.cols, mat.data, mat.isTranspose)
  }

  /*
  def toRowPartitionedMatrix(mat: IndexedRowMatrix) = {
    def densify(partition: Iterator[IndexedRow]) : Iterator[BDM[Double]] = {
      var block : BDM[Double] = BDM.zeros[Double](partition.length, mat.numCols.toInt)
      var currow = 0
      while(partition.hasNext) {
        block(currow, ::) := partition.next.vector.toBreeze.asInstanceOf[BDV[Double]].t
        currow += 1 
      }
      List(block).toIterator
    }
    RowPartitionedMatrix.fromMatrix(mat.rows.mapPartitions(densify))
  }
  */

  def report(message: String, verbose: Boolean = false) = {
    if(verbose) {
      println(message)
    }
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

  def sampledIdentityColumns(n : Int, numsamps: Int, probs: BDV[Double]) = {
    val rng = new java.util.Random
    val observations = List.fill(numsamps)(rng.nextDouble)
    val colCumProbs = accumulate(probs).toArray.zipWithIndex
    val keepIndices = observations.map( u => colCumProbs.find(_._1 >= u).getOrElse(Tuple2(1.0, n))._2.asInstanceOf[Int] ).toArray
    val omega = BDM.zeros[Double](n, numsamps)
    for( j <- 0 until numsamps) {
      omega(keepIndices(j), j) = 1  
    }
    omega
  }

  def sampleColumns(mat: IndexedRowMatrix, numcols: Int, probs: BDV[Double]) = {
    val rng = new java.util.Random
    val observations = List.fill(numcols)(rng.nextDouble)
    val colCumProbs = accumulate(probs).toArray.zipWithIndex
    val keepIndices = observations.map( u => colCumProbs.find(_._2 >= u).getOrElse(Tuple2(mat.numCols.toInt, 1.0))._1 )

    def subsampleVector(v : Vector) = {
      v match {
        case v: SparseVector => {
          val indexValPairs = (v.indices zip v.values) filter (pair => keepIndices contains pair._1)
          val (indices, values) = indexValPairs.unzip
          new SparseVector(numcols, indices.toArray, values.toArray)
        }
        case _ => throw new java.lang.IllegalArgumentException("expecting a SparseVector, got something else")
      }
    }
    new IndexedRowMatrix(mat.rows.map(row => IndexedRow(row.index, subsampleVector(row.vector))), mat.numRows, numcols)
  }

  // Returns `(mat.transpose * mat - avg*avg.transpose) * rhs`
  def multiplyCenteredGramianBy(mat: IndexedRowMatrix, rhs: DenseMatrix, avg: BDV[Double]): DenseMatrix = {
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
    fromBreeze(1.0/mat.numRows * result - avg * tmp.t)
  }

  /* get low-rank approximation to the input matrix using randomized PCA
   * algorithm. The rows of mat are the observations, 
   */
  def computeRPCA(mat: IndexedRowMatrix, rank: Int, slack: Int, numIters: Int,
    verbose: Boolean = false) : Tuple3[BDV[Double], BDM[Double], BDV[Double]] = {
    val rng = new java.util.Random
    val k = rank + slack
    var Y = DenseMatrix.randn(mat.numCols.toInt, k, rng).toBreeze.asInstanceOf[BDM[Double]]
    val rowavg = getRowMean(mat)

    report("performing iterations", verbose)
    for(i <- 0 until numIters) {
      Y = multiplyCenteredGramianBy(mat, fromBreeze(Y), rowavg).toBreeze.asInstanceOf[BDM[Double]]
      Y = qr.reduced.justQ(Y)
    }
    report("done iterations", verbose)

    report("performing QR to find basis", verbose)
    val Q = qr.reduced.justQ(Y)
    report("done performing QR", verbose)

    /* get the approximate top PCs by taking the SVD of an implicitly formed 
     * low-rank approximation to the covariance matrix
     * Q*Q.T * Cov * Q*Q.t
     */
    report("performing EVD", verbose)
    // manual resymmetrization should not be necessary with the next release of Breeze:
    // the current one is too sensitive, so you have to do this
    var B = 0.5 * Q.t * multiplyCenteredGramianBy(mat, fromBreeze(Q), rowavg).toBreeze.asInstanceOf[BDM[Double]] 
    B += B.t
    // NB: eigSym(B) gives U and d such that U*diag(d)*U^T = B, and the entries of d are in order of smallest to largest
    // so d is in the reverse of the usual mathematical order
    val eigSym.EigSym(lambdatilde, utilde) = eigSym(B)
    report("done performing EVD", verbose)
    val lambda = lambdatilde(k-1 until k-1-rank by -1)
    val U = (Q*utilde).apply(::, k-1 until k-1-rank by -1).copy

    (lambda, U, rowavg)
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
    val nparts = if(args.length >= 9) { args(8).toInt } else { 0 }
    val mat: IndexedRowMatrix = loadMSIData(sc, matkind, shape, inpath, nparts)
    mat.rows.cache()

    // force materialization of the RDD so we can separate I/O from compute
    mat.rows.count()

    // do the RPCA
    val rpcaResults = computeRPCA(mat, rank, slack, numIters, true)
    val rpcaEvals = rpcaResults._1
    val rpcaEvecs = rpcaResults._2
    val mean = rpcaResults._3

    //compute the CX (on centered data, this just uses the evecs from RPCA)
    val levProbs = sum(rpcaEvecs :^ 2.0, Axis._1) / rank.toDouble
    val sampleMat = sampledIdentityColumns(mat.numCols.toInt, rank, levProbs)
    val colsMat = multiplyCenteredGramianBy(mat, fromBreeze(sampleMat), mean).toBreeze.asInstanceOf[BDM[Double]]
    val qr.QR(q, r) = qr.reduced(colsMat)
    val cxQ = q
    val xMat = r \ multiplyCenteredGramianBy(mat, fromBreeze(cxQ), mean).toBreeze.asInstanceOf[BDM[Double]].t 

    //do the PCA
    val tol = 1e-10
    val maxIter = 300
    val covOperator = ( v: BDV[Double] ) =>  multiplyCenteredGramianBy(mat, fromBreeze(v.toDenseMatrix).transpose, mean).toBreeze.asInstanceOf[BDM[Double]].toDenseVector
    val (lambda2, u2) = EigenValueDecomposition.symmetricEigs(covOperator, mat.numCols.toInt, rank, tol, maxIter)
    val pcaEvals = lambda2
    val pcaEvecs = u2

    // estimate the relative Frobenius norm reconstruction errors
    val rng = new java.util.Random
    val numSamps = 1000
    val omega = DenseMatrix.randn(mat.numCols.toInt, numSamps, rng).toBreeze.asInstanceOf[BDM[Double]]
    var m = multiplyCenteredGramianBy(mat, fromBreeze(omega), mean).toBreeze.asInstanceOf[BDM[Double]] 
    val estFrobNorm = sqrt(1.0/numSamps * sum(m :* m))

    // estimate Frobenius norm approximation errors
    var diff = m - rpcaEvecs*(diag(rpcaEvals)*rpcaEvecs.t*omega)
    val rpcaEstFrobNormErr = sqrt(1.0/numSamps * sum(diff :* diff))
    diff = m - pcaEvecs*(diag(pcaEvals)*pcaEvecs.t*omega)
    val pcaEstFrobNormErr = sqrt(1.0/numSamps * sum(diff :* diff))
    diff = m - colsMat * (xMat * omega)
//    diff = m - cxQ * (multiplyCenteredGramianBy(mat, fromBreeze(cxQ), mean).toBreeze.asInstanceOf[BDM[Double]].t * omega)
    val cxEstFrobNormErr = sqrt(1.0/numSamps * sum(diff :* diff))

    println(f"RPCA estimated relative frobenius norm err ${rpcaEstFrobNormErr/estFrobNorm}")
    println(f"PCA estimated relative frobenius norm err ${pcaEstFrobNormErr/estFrobNorm}")
    println(f"CX estimated relative frobenius norm err ${cxEstFrobNormErr/estFrobNorm}")
//    println(f"CX uncentered relative frobenius norm err ${cxEstUncenteredFrobNormErr/estUncenteredFrobNorm}")
    
    // write output 
    val outf = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(new File(outpath))))
    outf.writeInt(shape._1)
    outf.writeInt(shape._2)
    outf.writeInt(rank)
    outf.writeInt(slack)
    outf.writeInt(numIters)
    outf.writeInt(nparts)
    outf.writeDouble(estFrobNorm)
    outf.writeDouble(rpcaEstFrobNormErr)
    outf.writeDouble(pcaEstFrobNormErr)
    outf.writeDouble(cxEstFrobNormErr)
    dumpV(outf, mean)
    dumpV(outf, rpcaEvals)
    dumpMat(outf, rpcaEvecs)
    dumpV(outf, pcaEvals)
    dumpMat(outf, pcaEvecs)
    dumpMat(outf, colsMat)
    dumpMat(outf, xMat)
    outf.close()
  }

  def dumpMat(outf: DataOutputStream, mat: BDM[Double]) = {
    outf.writeInt(mat.rows)
    outf.writeInt(mat.cols)
    for(i <- 0 until mat.rows) {
      for(j <- 0 until mat.cols) {
        outf.writeDouble(mat(i,j))
      }
    }
  }

  def dumpV(outf: DataOutputStream, v: BDV[Double]) = {
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

  def testEVD(sc: SparkContext) = {
    val tol = 1e-10
    val maxIter = 300
    val rank = 10
    var testmat = DenseMatrix.ones(50,50).toBreeze.asInstanceOf[BDM[Double]]
    testmat *= testmat.t

    def covOperator(v : BDV[Double]) :BDV[Double] = { testmat*v }
    val (lambda2, u2) = EigenValueDecomposition.symmetricEigs(covOperator, 50, rank, tol, maxIter)

    println(lambda2)
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("testEVD")
    conf.set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)

    appMain(sc, args)
  }

}
