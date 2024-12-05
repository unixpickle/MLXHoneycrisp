import HCTestUtils
import Honeycrisp
import XCTest

@testable import MLXHoneycrisp

final class MLXHoneycrispTests: XCTestCase {

  override func setUp() {
    Backend.defaultBackend = MLXBackend()
  }

  func testCast() async throws {
    try await BackendTests.testCast()
  }

  func testAdd() async throws {
    try await BackendTests.testAdd()
  }

  func testSub() async throws {
    try await BackendTests.testSub()
  }

  func testMul() async throws {
    try await BackendTests.testMul()
  }

  func testDiv() async throws {
    try await BackendTests.testDiv()
  }

  func testFusedAddMul() async throws {
    try await BackendTests.testFusedAddMul()
  }

  func testFusedNormalize() async throws {
    try await BackendTests.testFusedNormalize()
  }

  func testMod() async throws {
    try await BackendTests.testMod()
  }

  func testMulGrad() async throws {
    try await BackendTests.testMulGrad()
  }

  func testMSEGrad() async throws {
    try await BackendTests.testMSEGrad()
  }

  func testEquals() async throws {
    try await BackendTests.testEquals()
  }

  func testComparison() async throws {
    try await BackendTests.testComparison()
  }

  func testBitwise() async throws {
    try await BackendTests.testBitwise()
  }

  func testSum() async throws {
    try await BackendTests.testSum()
  }

  func testRepeat() async throws {
    try await BackendTests.testRepeat()
  }

  func testGather() async throws {
    try await BackendTests.testGather()
  }

  func testMatrixMatrixProduct() async throws {
    try await BackendTests.testMatrixMatrixProduct()
  }

  func testBatchedMatmul() async throws {
    try await BackendTests.testBatchedMatmul()
  }

  func testMatrixVectorProduct() async throws {
    try await BackendTests.testMatrixVectorProduct()
  }

  func testTril() async throws {
    try await BackendTests.testTril()
  }

  func testIndexing() async throws {
    try await BackendTests.testIndexing()
  }

  func testChunk() async throws {
    try await BackendTests.testChunk()
  }

  func testElemwise() async throws {
    try await BackendTests.testElemwise()
  }

  func testMinMax() async throws {
    try await BackendTests.testMinMax()
  }

  func testExpandAndRepeat() async throws {
    try await BackendTests.testExpandAndRepeat()
  }

  func testBinaryBroadcast() async throws {
    try await BackendTests.testBinaryBroadcast()
  }

  func testFusedBroadcast() async throws {
    try await BackendTests.testFusedBroadcast()
  }

  func testSoftmax() async throws {
    try await BackendTests.testSoftmax()
  }

  func testConcatInner() async throws {
    try await BackendTests.testConcatInner()
  }

  func testConcatOuter() async throws {
    try await BackendTests.testConcatOuter()
  }

  func testWhen() async throws {
    try await BackendTests.testWhen()
  }

  func testOneHot() async throws {
    try await BackendTests.testOneHot()
  }

  func testAdam() async throws {
    try await BackendTests.testAdam()
  }

  func testRandom() async throws {
    try await BackendTests.testRandom()
  }

  func testConv2D() async throws {
    try await BackendTests.testConv2D()
  }

  func testConv2DTransposeGrads() async throws {
    try await BackendTests.testConv2DTransposeGrads()
  }

  func testConv1D() async throws {
    try await BackendTests.testConv1D()
  }

  func testGroupNorm() async throws {
    try await BackendTests.testGroupNorm()
  }

}
