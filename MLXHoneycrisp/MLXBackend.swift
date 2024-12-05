import Accelerate
import Foundation
import HCBacktrace
import Honeycrisp
import MLX
import MLXFast
import MLXNN
import MLXRandom

extension DType {
  var byteSize: Int {
    switch self {
    case .bool:
      1
    case .uint8:
      1
    case .float16:
      2
    case .float32:
      4
    case .uint32:
      4
    case .int64:
      8
    default:
      tracedFatalError("unsupported data type: \(self)")
    }
  }
}

public class MLXBackend: CPUBackend {

  public class MLXRandomGenerator: RandomGenerator {
    public let mlxBackend: MLXBackend

    override open var stateCount: Int {
      1
    }

    override open var stateDType: Tensor.DType {
      .int64
    }

    init(mlxBackend: MLXBackend, seed: Int) {
      self.mlxBackend = mlxBackend
      super.init(
        backend: mlxBackend,
        state: Tensor(
          dataTask: Tensor.createDataTask {
            try await mlxBackend.serialize {
              MLXData(backend: mlxBackend, data: MLXRandom.key(UInt64(seed)))
            }
          },
          shape: [1],
          dtype: .int64
        )
      )
    }

    override open func _seed(_ x: Int) async throws -> Tensor.Data {
      let mlxBackend = mlxBackend
      return try await mlxBackend.serialize {
        using(device: mlxBackend.device) {
          MLXData(backend: mlxBackend, data: MLXRandom.key(UInt64(x)))
        }
      }
    }

    override open func _sample(
      state: Tensor.Data, count: Int, dist: RandomDist, dtype: Tensor.DType
    )
      async throws -> (
        sample: Tensor.Data, state: Tensor.Data
      )
    {
      let mlxBackend = mlxBackend
      let dtype = mlxBackend.mlxDType(dtype)
      let oldKey = try await mlxBackend.asArray(state, 2, .uint32)
      return try await mlxBackend.serialize {
        using(device: mlxBackend.device) {
          let keys = MLXRandom.split(key: oldKey, into: 2)
          let newKey = keys[0]
          let useKey = keys[1]
          let results =
            switch dist {
            case .normal:
              MLXRandom.normal(
                [count],
                dtype: dtype,
                loc: 0,
                scale: 1.0,
                key: useKey
              )
            case .uniform:
              MLXRandom.uniform(
                low: 0.0,
                high: 1.0,
                [count],
                dtype: dtype,
                key: useKey
              )
            }
          return (
            sample: MLXData(
              backend: mlxBackend,
              data: results
            ),
            state: MLXData(backend: mlxBackend, data: newKey)
          )
        }
      }
    }

    override open func _sample(state: Tensor.Data, count: Int, in range: Range<Int64>)
      async throws -> (
        sample: Tensor.Data, state: Tensor.Data
      )
    {
      let mlxBackend = mlxBackend
      let oldKey = try await mlxBackend.asArray(state, 2, .uint32)
      return try await mlxBackend.serialize {
        using(device: mlxBackend.device) {
          let keys = MLXRandom.split(key: oldKey, into: 2)
          let newKey = keys[0]
          let useKey = keys[1]
          return (
            sample: MLXData(
              backend: mlxBackend,
              data: MLXRandom.randInt(
                low: range.lowerBound,
                high: range.upperBound,
                [count],
                type: Int64.self,
                key: useKey
              )
            ),
            state: MLXData(backend: mlxBackend, data: newKey)
          )
        }
      }
    }
  }

  public let device: Device
  private var _defaultRandomMLX: MLXRandomGenerator? = nil

  public init(device: Device = .gpu) {
    self.device = device
    super.init()
    self._defaultRandomMLX = MLXRandomGenerator(
      mlxBackend: self, seed: Int.random(in: 0..<1_000_000_000))
  }

  public class MLXData: Tensor.Data {
    private let backend: MLXBackend
    private var data: MLXArray

    public init(backend: MLXBackend, byteCount: Int) async throws {
      self.backend = backend
      data = try await backend.serialize {
        MLXArray.zeros([byteCount], type: UInt8.self, stream: .device(backend.device))
      }
    }

    public init(backend: MLXBackend, data: MLXArray) {
      alwaysAssert(data.shape.count == 1, "data shape \(data.shape) should be one-dimensional")
      self.backend = backend
      self.data = data
    }

    public var byteCount: Int {
      data.shape[0] * data.dtype.byteSize
    }

    public func dataAsType(_ dtype: DType) async throws -> MLXArray {
      if dtype == data.dtype {
        data
      } else {
        try await backend.serialize { [self] in
          data.view(dtype: dtype, stream: .device(backend.device))
        }
      }
    }

    public func onCPU<T>(_ fn: (_: UnsafeRawPointer) async throws -> T) async throws -> T {
      let ptrResult = try await backend.serialize { [self] in
        data.rawPointer()
      }
      defer { ptrResult.dealloc() }
      return try await fn(ptrResult.ptr)
    }

    public func mutateOnCPU<T>(_ fn: (_: UnsafeMutableRawPointer) async throws -> T) async throws
      -> T
    {
      let ptrResult = try await backend.serialize { [self] in
        data.rawPointer()
      }
      defer { ptrResult.dealloc() }
      alwaysAssert(!ptrResult.isCopy, "cannot mutate a discontiguous tensor")
      return try await fn(ptrResult.ptr)
    }
  }

  override open func allocate(_ byteCount: Int) async throws -> Tensor.Data {
    try await MLXData(backend: self, byteCount: byteCount)
  }

  internal func mlxDType(_ dtype: Tensor.DType) -> DType {
    switch dtype {
    case .float32:
      .float32
    case .float16:
      .float16
    case .bool:
      .bool
    case .int64:
      .int64
    }
  }

  internal func asArray(_ data: Tensor.Data, _ count: Int, _ dtype: DType) async throws -> MLXArray
  {
    if let mlxData = data as? MLXData {
      let device = device
      let result = try await mlxData.dataAsType(dtype)
      alwaysAssert(result.shape[0] >= count, "array size \(result.shape) must contain \(count)")
      if result.shape[0] == count {
        return result
      } else {
        return try await serialize {
          using(device: device) {
            result[0..<count]
          }
        }
      }
    }

    let device = device
    alwaysAssert(
      data.byteCount >= count * dtype.byteSize,
      "data size \(data.byteCount) must be at least \(count*dtype.byteSize) bytes")
    return try await data.onCPU { buffer in
      try await serialize {
        MLXArray(
          UnsafeRawBufferPointer(start: buffer, count: dtype.byteSize * count), [data.byteCount],
          type: UInt8.self
        ).view(dtype: dtype, stream: .device(device))
      }
    }
  }

  internal func broadcastShapes(_ a: BroadcastStrides, _ b: BroadcastStrides) -> ([Int], [Int]) {
    let results = broadcastShapes([a, b])
    return (results[0], results[1])
  }

  internal func broadcastShapes(_ a: BroadcastStrides, _ b: BroadcastStrides, _ c: BroadcastStrides)
    -> ([Int], [Int], [Int])
  {
    let results = broadcastShapes([a, b, c])
    return (results[0], results[1], results[2])
  }

  internal func broadcastShapes(
    _ a: BroadcastStrides, _ b: BroadcastStrides, _ c: BroadcastStrides, _ d: BroadcastStrides
  )
    -> ([Int], [Int], [Int], [Int])
  {
    let results = broadcastShapes([a, b, c, d])
    return (results[0], results[1], results[2], results[3])
  }

  internal func broadcastShapes(_ inStrides: [BroadcastStrides]) -> [[Int]] {
    var shapes = inStrides.map { $0.shape }
    var strides = inStrides.map { $0.strides }
    var i = shapes[0].count - 1
    while i > 0 {
      var allCollapse = true
      for (shape, strides) in zip(shapes, strides) {
        if !(strides[i] == 0 && strides[i - 1] == 0)
          && !(strides[i] != 0 && strides[i - 1] != 0 && (strides[i - 1] == shape[i] * strides[i]))
        {
          allCollapse = false
        }
      }
      if allCollapse {
        for j in 0..<shapes.count {
          shapes[j][i - 1] *= shapes[j][i]
          shapes[j].remove(at: i)
          strides[j][i - 1] = strides[j][i]
          strides[j].remove(at: i)
        }
      }
      i -= 1
    }
    return zip(shapes, strides).map { shape, stride in
      zip(shape, stride).map { size, stride in stride == 0 ? 1 : size }
    }
  }

  override open func constant<T: TensorElement>(_ value: T, count: Int, dtype: Tensor.DType)
    async throws -> Tensor.Data
  {
    let result = try await serialize { [self] in
      using(device: device) {
        let scalar =
          if T.isFloatLossy {
            MLXArray(value.toInt64()).asType(mlxDType(dtype))
          } else {
            MLXArray(value.toFloat()).asType(mlxDType(dtype))
          }
        return MLXArray.full([count], values: scalar)
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func collection<T: TensorElement>(
    _ collection: some Collection<T>, reverse: Bool, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    let result = try await serialize { [self] in
      let arr = (reverse ? collection.reversed() : Array(collection))
      return using(device: device) {
        let result =
          if T.isFloatLossy {
            MLXArray(arr.map { $0.toInt64() })
          } else {
            MLXArray(arr.map { $0.toFloat() })
          }
        return result.asType(mlxDType(dtype))
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data {
    let device = device
    let result = try await serialize {
      using(device: device) {
        let oldStrides = stridesForShape(shape)
        let oneShape = Array(repeating: 1, count: permutation.count)
        let allIndices = MLXArray.zeros(oneShape, dtype: .int64)
        for (axis, sourceAxis) in permutation.enumerated() {
          let oldStride = oldStrides[sourceAxis]
          let oldSize = shape[sourceAxis]
          var shape = oneShape
          shape[axis] = oldSize
          allIndices += (oldStride * MLXArray(Int64(0)..<Int64(oldSize))).reshaped(
            shape)
        }
        return allIndices.flattened()
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func broadcast(_ a: BroadcastData, dtype: Tensor.DType) async throws -> Tensor.Data
  {
    let arr1 = try await asArray(a.data, a.dataCount, mlxDType(dtype))

    let device = device
    let arrOut = try await serialize {
      using(device: device) {
        MLX.broadcast(arr1.reshaped(a.strides.dataShape), to: a.strides.shape).flattened()
      }
    }
    return MLXData(backend: self, data: arrOut)
  }

  override open func binaryOp(
    _ a: BroadcastData, _ b: BroadcastData, op: NumericBinaryOp, dtype: Tensor.DType
  ) async throws
    -> Tensor.Data
  {
    let (aShape, bShape) = broadcastShapes(a.strides, b.strides)
    let arr1 = try await asArray(a.data, a.dataCount, mlxDType(dtype))
    let arr2 = try await asArray(b.data, b.dataCount, mlxDType(dtype))

    let device = device
    let arrOut = try await serialize {
      using(device: device) {
        let arr1 = arr1.reshaped(aShape)
        let arr2 = arr2.reshaped(bShape)
        let result =
          switch op {
          case .add:
            arr1 + arr2
          case .sub:
            arr1 - arr2
          case .div:
            dtype.isFloat ? arr1 / arr2 : arr1.floorDivide(arr2)
          case .mul:
            arr1 * arr2
          case .mod:
            arr1 % arr2
          }
        return result.flattened()
      }
    }
    return MLXData(backend: self, data: arrOut)
  }

  override open func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  ) async throws
    -> Tensor.Data
  {
    let arr1 = try await asArray(a, count, mlxDType(dtype))

    let device = device
    func apply<T1: HasDType>(_ b: T1) async throws -> MLXArray {
      try await serialize {
        using(device: device) {
          switch op {
          case .add:
            arr1 + b
          case .sub:
            arr1 - b
          case .div:
            dtype.isFloat ? arr1 / b : arr1.floorDivide(b)
          case .mul:
            arr1 * b
          case .mod:
            arr1 % b
          }
        }
      }
    }

    let arrOut =
      switch dtype {
      case .float16:
        try await apply(Float16(b.toFloat()))
      case .float32:
        try await apply(b.toFloat())
      case .int64:
        try await apply(b.toInt64())
      default:
        tracedFatalError("unsupported dtype: \(dtype)")
      }

    return MLXData(backend: self, data: arrOut)
  }

  override open func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  ) async throws
    -> Tensor.Data
  {
    let mDtype = mlxDType(dtype)
    let arr1 = try await asArray(b, count, mDtype)

    let device = device
    func apply<T1: HasDType>(_ a: T1) async throws -> MLXArray {
      try await serialize {
        using(device: device) {
          switch op {
          case .add:
            a + arr1
          case .sub:
            a - arr1
          case .div:
            dtype.isFloat ? a / arr1 : MLXArray(a).asType(mDtype).floorDivide(arr1)
          case .mul:
            a * arr1
          case .mod:
            a % arr1
          }
        }
      }
    }

    let arrOut =
      switch dtype {
      case .float16:
        try await apply(Float16(a.toFloat()))
      case .float32:
        try await apply(a.toFloat())
      case .int64:
        try await apply(a.toInt64())
      default:
        tracedFatalError("unsupported dtype: \(dtype)")
      }

    return MLXData(backend: self, data: arrOut)
  }

  override open func compare(
    _ a: BroadcastData, _ b: BroadcastData, op: ComparisonOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let (aShape, bShape) = broadcastShapes(a.strides, b.strides)
    let arr1 = try await asArray(a.data, a.dataCount, mlxDType(dtype))
    let arr2 = try await asArray(b.data, b.dataCount, mlxDType(dtype))
    let device = device
    let result: MLXArray = try await serialize {
      using(device: device) {
        let arr1 = arr1.reshaped(aShape)
        let arr2 = arr2.reshaped(bShape)
        let result: MLXArray =
          switch op {
          case .equal:
            arr1 .== arr2
          case .less:
            arr1 .< arr2
          case .lessEqual:
            arr1 .<= arr2
          case .greater:
            arr1 .> arr2
          case .greaterEqual:
            arr1 .>= arr2
          }
        return result.flattened()
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func compare<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let arr1 = try await asArray(a, count, mlxDType(dtype))
    let device = device
    let result: MLXArray = try await serialize {
      func apply<T1: ScalarOrArray>(_ b: T1) -> MLXArray {
        using(device: device) {
          switch op {
          case .equal:
            arr1 .== b
          case .less:
            arr1 .< b
          case .lessEqual:
            arr1 .<= b
          case .greater:
            arr1 .> b
          case .greaterEqual:
            arr1 .>= b
          }
        }
      }
      if T.isFloatLossy {
        return apply(b.toInt64())
      } else {
        return apply(b.toFloat())
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func compare<T: TensorElement>(
    _ a: T, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let arr2 = try await asArray(b, count, mlxDType(dtype))
    let device = device
    let result: MLXArray = try await serialize {
      func apply<T1: ScalarOrArray>(_ a: T1) -> MLXArray {
        using(device: device) { () -> MLXArray in
          switch op {
          case .equal:
            arr2 .== a
          case .less:
            arr2 .> a
          case .lessEqual:
            arr2 .<= a
          case .greater:
            arr2 .< a
          case .greaterEqual:
            arr2 .<= a
          }
        }
      }
      if T.isFloatLossy {
        return apply(a.toInt64())
      } else {
        return apply(a.toFloat())
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func bitwiseOp(
    _ a: BroadcastData, _ b: BroadcastData, op: BitwiseOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let intType: DType =
      switch dtype.byteSize {
      case 1:
        .uint8
      case 2:
        .uint16
      case 4:
        .uint32
      case 8:
        .uint64
      default:
        fatalError("unsupported dtype size")
      }
    let resultDType = mlxDType(dtype)
    let x = try await asArray(a.data, a.dataCount, intType)
    let y = try await asArray(b.data, b.dataCount, intType)
    let (xShape, yShape) = broadcastShapes(a.strides, b.strides)
    let result = try await serialize { [self] in
      using(device: device) {
        let result =
          switch op {
          case .and:
            x.reshaped(xShape) & y.reshaped(yShape)
          case .or:
            x.reshaped(xShape) | y.reshaped(yShape)
          case .xor:
            x.reshaped(xShape) ^ y.reshaped(yShape)
          }
        return result.flattened().view(dtype: resultDType)
      }
    }
    return MLXData(backend: self, data: result)
  }

  /// Perform a bitwise operator between a tensor and a scalar.
  ///
  /// The `count` is the number of elements in the input and the output.
  override open func bitwiseOp<T: TensorElementBitPattern>(
    _ a: Tensor.Data, _ b: T, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let intType: DType =
      switch dtype.byteSize {
      case 1:
        .uint8
      case 2:
        .uint16
      case 4:
        .uint32
      case 8:
        .uint64
      default:
        fatalError("unsupported dtype size")
      }
    let resultDType = mlxDType(dtype)
    let x = try await asArray(a, count, intType)
    let result = try await serialize { [self] in
      using(device: device) {
        let y = MLXArray(b.bitsForBitwiseOp).view(dtype: intType)
        let result =
          switch op {
          case .and:
            x & y
          case .or:
            x | y
          case .xor:
            x ^ y
          }
        return result.view(dtype: resultDType)
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func cast(
    _ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let arr1 = try await asArray(a, count, mlxDType(inType))
    let device = device
    let result = try await serialize { [self] in
      using(device: device) {
        arr1.asType(mlxDType(outType))
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, scale: T, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let arr1 = try await asArray(a, count, mlxDType(dtype))
    let arr2: MLXArray? =
      if let scales = scales {
        try await asArray(scales, count, mlxDType(dtype))
      } else {
        nil
      }
    let device = device
    let result = try await serialize {
      using(device: device) {
        var result = arr1.pow(b.toFloat())
        if scale != T(1.0) {
          result = result * scale.toFloat()
        }
        if let arr2 = arr2 {
          result = result * arr2
        }
        return result
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func clamp<T: NumericTensorElement>(
    _ a: Tensor.Data, min: T?, max: T?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    alwaysAssert(min != nil || max != nil, "cannot use clamp() without bounds")
    let arr1 = try await asArray(a, count, mlxDType(dtype))
    let device = device
    let result: MLXArray = try await serialize {
      using(device: device) {
        if dtype == .int64 {
          if let max = max, min == nil {
            MLX.minimum(arr1, max.toInt64())
          } else if let min = min, max == nil {
            MLX.maximum(arr1, min.toInt64())
          } else {
            MLX.clip(arr1, min: min!.toInt64(), max: max!.toInt64())
          }
        } else {
          if let max = max, min == nil {
            MLX.minimum(arr1, max.toFloat())
          } else if let min = min, max == nil {
            MLX.maximum(arr1, min.toFloat())
          } else {
            MLX.clip(arr1, min: min!.toFloat(), max: max!.toFloat())
          }
        }
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func when<T>(
    _ mask: BroadcastData, _ a: TensorOrScalar<T>, _ b: TensorOrScalar<T>, _: T.Type,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    var strides = [mask.strides]
    for arg in [a, b] {
      if case .tensor(let t) = arg {
        strides.append(t.strides)
      }
    }
    let shapes = broadcastShapes(strides)
    let maskArr = try await asArray(mask.data, mask.dataCount, mlxDType(.bool))
    let (aArr, aShape) =
      switch a {
      case .tensor(let t):
        (try await asArray(t.data, t.dataCount, mlxDType(dtype)), shapes[1])
      case .scalar(let s, _):
        (
          try await serialize {
            T.isFloatLossy ? MLXArray(s.toInt64()) : MLXArray(s.toFloat())
          },
          []
        )
      }

    let (bArr, bShape) =
      switch b {
      case .tensor(let t):
        (try await asArray(t.data, t.dataCount, mlxDType(dtype)), shapes.last!)
      case .scalar(let s, _):
        (
          try await serialize {
            T.isFloatLossy ? MLXArray(s.toInt64()) : MLXArray(s.toFloat())
          },
          []
        )
      }

    let device = device
    let result: MLXArray = try await serialize { [self] in
      using(device: device) {
        MLX.which(
          maskArr.reshaped(shapes[0]),
          aArr.reshaped(aShape).asType(mlxDType(dtype)),
          bArr.reshaped(bShape).asType(mlxDType(dtype))
        ).flattened()
      }
    }
    return MLXData(backend: self, data: result)
  }

  internal struct ElemwiseKey: Equatable, Hashable {
    public let op: ElemwiseOp
    public let hasScales: Bool
    public let count: Int
    public let dtype: Tensor.DType
  }

  internal var elemwiseCache = [ElemwiseKey: @Sendable ([MLXArray]) -> [MLXArray]]()

  override open func elemwise(
    _ a: Tensor.Data, op: ElemwiseOp, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  ) async throws -> Tensor.Data {
    let arr1 = try await asArray(a, count, mlxDType(dtype))
    let arr2: MLXArray? =
      if let scales = scales {
        try await asArray(scales, count, mlxDType(dtype))
      } else {
        nil
      }
    let device = device
    let result: MLXArray = try await serialize { [self] in
      using(device: device) {
        let key = ElemwiseKey(
          op: op,
          hasScales: arr2 != nil,
          count: count,
          dtype: dtype
        )
        let fn =
          elemwiseCache[key]
          ?? MLX.compile { inputs in
            let arr1 = inputs[0].asType(.float32)
            let arr2: MLXArray? = inputs.count > 1 ? inputs[1].asType(.float32) : nil
            var result: MLXArray =
              switch op {
              case .sin:
                arr1.sin()
              case .cos:
                arr1.cos()
              case .minusSin:
                -arr1.sin()
              case .exp:
                arr1.exp()
              case .log:
                arr1.log()
              case .recip:
                1 / arr1
              case .sigmoid:
                MLXNN.sigmoid(arr1)
              case .sigmoidGrad:
                MLX.grad({ arg in [MLXNN.sigmoid(arg).sum()] })(arr1)[0]
              case .relu:
                MLXNN.relu(arr1)
              case .reluGrad:
                MLX.where(arr1 .< 0, MLXArray.zeros(like: arr1), MLXArray.ones(like: arr1))
              case .abs:
                arr1.abs()
              case .absGrad:
                MLX.where(arr1 .< 0, -MLXArray.ones(like: arr1), MLXArray.ones(like: arr1))
              case .gelu:
                MLXNN.geluApproximate(arr1)
              case .geluGrad:
                MLX.grad({ arg in [MLXNN.geluApproximate(arg).sum()] })(arr1)[0]
              }
            if let arr2 = arr2 {
              result = result * arr2
            }
            return [result.asType(inputs[0].dtype)]
          }
        if elemwiseCache[key] == nil {
          elemwiseCache[key] = fn
        }
        return fn(arr2 == nil ? [arr1] : [arr1, arr2!])[0]
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    var arrs = [MLXArray]()
    for (input, innerCount) in zip(inputs, innerCounts) {
      arrs.append(try await asArray(input, innerCount * outerCount, mlxDType(dtype)))
    }
    let device = device
    let result: MLXArray = try await serialize {
      using(device: device) {
        MLX.concatenated(
          zip(arrs, innerCounts).map { (arr, count) in arr.reshaped(outerCount, count) }, axis: 1
        ).flattened()
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func logSoftmax(
    _ a: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let arr1 = try await asArray(a, dims.inCount, mlxDType(dtype))
    let device = device
    let result: MLXArray = try await serialize {
      using(device: device) {
        MLXNN.logSoftmax(
          arr1.reshaped(dims.outerCount, dims.reduceCount, dims.innerCount).asType(.float32),
          axis: 1
        )
        .flattened().asType(arr1.dtype)
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let arr1 = try await asArray(a, dims.inCount, mlxDType(dtype))
    let arr2 = try await asArray(outGrad, dims.inCount, mlxDType(dtype))
    let device = device
    let result: MLXArray = try await serialize {
      using(device: device) {
        let grad = arr2.reshaped(dims.outerCount, dims.reduceCount, dims.innerCount).asType(
          .float32)
        return
          (grad
          - (MLX.softmax(
            arr1.reshaped(dims.outerCount, dims.reduceCount, dims.innerCount).asType(.float32),
            axis: 1) * grad.sum(axis: 1, keepDims: true)))
          .flattened().asType(arr1.dtype)
      }
    }
    return MLXData(backend: self, data: result)
  }

  internal func gatherScatterShapes(_ s: ScatterGatherIndices) -> (
    indexShape: [Int], inputShape: [Int], outputShape: [Int], axis: Int
  ) {
    let squashBefore = Set(s.indices.strides.strides[..<s.axis].map { $0 == 0 }).count == 1
    let squashAfter = Set(s.indices.strides.strides[(s.axis + 1)...].map { $0 == 0 }).count == 1
    let newAxis = squashBefore ? 1 : s.axis

    func maybeSquash(_ squash: Bool, _ dims: some Collection<Int>) -> [Int] {
      squash ? [dims.reduce(1, { $0 * $1 })] : Array(dims)
    }
    func squashShape(_ shape: [Int]) -> [Int] {
      (maybeSquash(squashBefore, shape[..<s.axis]) + [shape[s.axis]]
        + maybeSquash(squashAfter, shape[(s.axis + 1)...]))
    }

    return (
      indexShape: squashShape(s.indices.strides.dataShape),
      inputShape: squashShape(s.valueShape),
      outputShape: squashShape(s.indices.strides.shape),
      axis: newAxis
    )
  }

  override open func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let arr1 = try await asArray(a, s.gatherInCount, mlxDType(dtype))
    let arr2 = try await asArray(s.indices.data, s.indices.dataCount, .int64)
    let shapes = gatherScatterShapes(s)

    let device = device
    let result: MLXArray = try await serialize {
      using(device: device) {
        takeAlong(
          arr1.reshaped(shapes.inputShape),
          arr2.reshaped(shapes.indexShape),
          axis: shapes.axis
        ).flattened()
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let arr1 = try await asArray(a, s.gatherOutCount, mlxDType(dtype))
    let arr2 = try await asArray(s.indices.data, s.indices.dataCount, .int64)
    let shapes = gatherScatterShapes(s)

    let device = device
    let result: MLXArray = try await serialize {
      using(device: device) {
        if s.indicesAreUnique {
          let result = MLXArray.zeros(shapes.inputShape, dtype: arr1.dtype)
          return putAlong(
            result,
            MLX.broadcast(arr2.reshaped(shapes.indexShape), to: shapes.outputShape),
            values: arr1.reshaped(shapes.outputShape),
            axis: shapes.axis
          ).flattened()
        } else {
          let emptyInputs = MLXArray.zeros(
            [shapes.inputShape.reduce(1, { $0 * $1 })], dtype: arr1.dtype)
          return MLX.grad({ (inputs: MLXArray) in
            [
              (takeAlong(
                inputs.reshaped(shapes.inputShape),
                arr2.reshaped(shapes.indexShape),
                axis: shapes.axis
              ).flattened() * arr1).sum()
            ]
          })(emptyInputs)[0]
        }
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func repeated(
    _ a: Tensor.Data, dims: RepeatDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let arr1 = try await asArray(a, dims.inCount, mlxDType(dtype))
    let device = device
    let result: MLXArray = try await serialize {
      using(device: device) {
        MLX.broadcast(
          arr1.reshaped(dims.outerCount, 1, dims.innerCount),
          to: [dims.outerCount, dims.repeatCount, dims.innerCount]
        ).flattened()
      }
    }
    return MLXData(backend: self, data: result)
  }

  override open func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let arr1 = try await asArray(a, dims.inCount, mlxDType(dtype))
    let device = device
    let result: MLXArray = try await serialize {
      using(device: device) {
        let input = arr1.reshaped(dims.outerCount, dims.reduceCount, dims.innerCount)
        let result =
          switch op {
          case .sum:
            input.sum(axis: 1)
          case .argmax:
            input.argMax(axis: 1).asType(.int64)
          case .argmin:
            input.argMin(axis: 1).asType(.int64)
          }
        return result.flattened()
      }
    }
    return MLXData(backend: self, data: result)
  }

  internal struct NormalizeKey: Equatable, Hashable {
    public let input: [Int]
    public let mean: [Int]
    public let variance: [Int]
    public let epsilon: Float
    public let dtype: Tensor.DType
  }

  internal var normalizeCache = [NormalizeKey: @Sendable ([MLXArray]) -> [MLXArray]]()

  override open func normalize<T: TensorElement>(
    input: BroadcastData, mean: BroadcastData, variance: BroadcastData, epsilon: T,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    assert(dtype.isFloat)
    let (inputShape, meanShape, varShape) = broadcastShapes(
      input.strides, mean.strides, variance.strides)
    let inputArr = try await asArray(input.data, input.dataCount, mlxDType(dtype))
    let meanArr = try await asArray(mean.data, mean.dataCount, mlxDType(dtype))
    let varArr = try await asArray(variance.data, variance.dataCount, mlxDType(dtype))

    let device = device
    let arrOut = try await serialize { [self] in
      using(device: device) {
        let key = NormalizeKey(
          input: inputShape,
          mean: meanShape,
          variance: varShape,
          epsilon: epsilon.toFloat(),
          dtype: dtype
        )
        let fn =
          normalizeCache[key]
          ?? MLX.compile { inputs in
            let inputArr = inputs[0]
            let meanArr = inputs[1]
            let varArr = inputs[2]
            return
              [
                ((inputArr.reshaped(inputShape) - meanArr.reshaped(meanShape))
                  * (varArr.reshaped(varShape) + epsilon.toFloat()).rsqrt()).flattened()
              ]
          }
        if normalizeCache[key] == nil {
          normalizeCache[key] = fn
        }
        return fn([inputArr, meanArr, varArr])[0]
      }
    }
    return MLXData(backend: self, data: arrOut)
  }

  internal struct NormalizeXGradKey: Equatable, Hashable {
    public let variance: [Int]
    public let outGrad: [Int]
    public let epsilon: Float
    public let sign: Float
    public let dtype: Tensor.DType
  }

  internal var normalizeXGradCache = [NormalizeXGradKey: @Sendable ([MLXArray]) -> [MLXArray]]()

  override open func normalizeXGrad<T: TensorElement>(
    variance: BroadcastData, outGrad: BroadcastData, epsilon: T, sign: Float,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    assert(dtype.isFloat)
    let (varShape, outGradShape) = broadcastShapes(variance.strides, outGrad.strides)
    let varArr = try await asArray(variance.data, variance.dataCount, mlxDType(dtype))
    let outGradArr = try await asArray(outGrad.data, outGrad.dataCount, mlxDType(dtype))

    let device = device
    let arrOut = try await serialize { [self] in
      using(device: device) {
        let key = NormalizeXGradKey(
          variance: varShape,
          outGrad: outGradShape,
          epsilon: epsilon.toFloat(),
          sign: sign,
          dtype: dtype
        )
        let fn =
          normalizeXGradCache[key]
          ?? MLX.compile { inputs in
            let varArr = inputs[0]
            let outGradArr = inputs[1]
            return
              [
                (outGradArr.reshaped(outGradShape)
                  * (varArr.reshaped(varShape) + epsilon.toFloat()).rsqrt() * sign).flattened()
              ]
          }
        if normalizeXGradCache[key] == nil {
          normalizeXGradCache[key] = fn
        }
        return fn([varArr, outGradArr])[0]

      }
    }
    return MLXData(backend: self, data: arrOut)
  }

  internal struct NormalizeVarGradKey: Equatable, Hashable {
    public let input: [Int]
    public let mean: [Int]
    public let variance: [Int]
    public let outGrad: [Int]
    public let epsilon: Float
    public let dtype: Tensor.DType
  }

  internal var normalizeVarGradCache = [NormalizeVarGradKey: @Sendable ([MLXArray]) -> [MLXArray]]()

  override open func normalizeVarianceGrad<T: TensorElement>(
    input: BroadcastData, mean: BroadcastData, variance: BroadcastData, outGrad: BroadcastData,
    epsilon: T, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    assert(dtype.isFloat)
    let (inputShape, meanShape, varShape, outGradShape) = broadcastShapes(
      input.strides, mean.strides, variance.strides, outGrad.strides)
    let inputArr = try await asArray(input.data, input.dataCount, mlxDType(dtype))
    let meanArr = try await asArray(mean.data, mean.dataCount, mlxDType(dtype))
    let varArr = try await asArray(variance.data, variance.dataCount, mlxDType(dtype))
    let outGradArr = try await asArray(outGrad.data, outGrad.dataCount, mlxDType(dtype))

    let device = device
    let arrOut = try await serialize { [self] in
      using(device: device) {
        let key = NormalizeVarGradKey(
          input: inputShape,
          mean: meanShape,
          variance: varShape,
          outGrad: outGradShape,
          epsilon: epsilon.toFloat(),
          dtype: dtype
        )
        let fn =
          normalizeVarGradCache[key]
          ?? MLX.compile { inputs in
            let inputArr = inputs[0]
            let meanArr = inputs[1]
            let varArr = inputs[2]
            let outGradArr = inputs[3]
            return
              [
                (-0.5 * outGradArr.reshaped(outGradShape).asType(.float32)
                  * (inputArr.reshaped(inputShape).asType(.float32)
                    - meanArr.reshaped(meanShape).asType(.float32))
                  * MLX.pow(varArr.reshaped(varShape).asType(.float32) + epsilon.toFloat(), -1.5))
                  .flattened().asType(inputArr.dtype)
              ]
          }
        if normalizeVarGradCache[key] == nil {
          normalizeVarGradCache[key] = fn
        }
        return fn([inputArr, meanArr, varArr, outGradArr])[0]
      }
    }
    return MLXData(backend: self, data: arrOut)
  }

  override open func mulAdd(
    input: BroadcastData, coeff: BroadcastData, bias: BroadcastData, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let (inputShape, coeffShape, biasShape) = broadcastShapes(
      input.strides, coeff.strides, bias.strides)
    let inputArr = try await asArray(input.data, input.dataCount, mlxDType(dtype))
    let coeffArr = try await asArray(coeff.data, coeff.dataCount, mlxDType(dtype))
    let biasArr = try await asArray(bias.data, bias.dataCount, mlxDType(dtype))

    let device = device
    let arrOut = try await serialize {
      using(device: device) {
        (inputArr.reshaped(inputShape) * coeffArr.reshaped(coeffShape) + biasArr.reshaped(biasShape))
          .flattened()
      }
    }
    return MLXData(backend: self, data: arrOut)
  }

  override open func addMul(
    input: BroadcastData, bias: BroadcastData, coeff: BroadcastData, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let (inputShape, coeffShape, biasShape) = broadcastShapes(
      input.strides, coeff.strides, bias.strides)
    let inputArr = try await asArray(input.data, input.dataCount, mlxDType(dtype))
    let coeffArr = try await asArray(coeff.data, coeff.dataCount, mlxDType(dtype))
    let biasArr = try await asArray(bias.data, bias.dataCount, mlxDType(dtype))

    let device = device
    let arrOut = try await serialize {
      using(device: device) {
        ((inputArr.reshaped(inputShape) + biasArr.reshaped(biasShape))
          * coeffArr.reshaped(coeffShape)).flattened()
      }
    }
    return MLXData(backend: self, data: arrOut)
  }

  internal struct MatmulKey: Equatable, Hashable {
    public let matrixCount: Int
    public let transA: Bool
    public let transB: Bool
    public let rows: Int
    public let inner: Int
    public let cols: Int
    public let dtype: Tensor.DType
  }

  internal var matmulCache = [MatmulKey: @Sendable ([MLXArray]) -> [MLXArray]]()

  override open func batchedMatmul(
    matrixCount: Int, a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool,
    rows: Int, inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    if transOut {
      return try await batchedMatmul(
        matrixCount: matrixCount, a: b, transA: !transB, b: a, transB: !transA, transOut: false,
        rows: cols, inner: inner, cols: rows, dtype: dtype)
    }

    let arrA = try await asArray(a, matrixCount * rows * inner, mlxDType(dtype))
    let arrB = try await asArray(b, matrixCount * inner * cols, mlxDType(dtype))
    let device = device
    let result = try await serialize { [self] in
      using(device: device) {
        let key = MatmulKey(
          matrixCount: matrixCount,
          transA: transA,
          transB: transB,
          rows: rows,
          inner: inner,
          cols: cols,
          dtype: dtype
        )
        let fn =
          matmulCache[key]
          ?? MLX.compile { (args: [MLXArray]) in
            let arrA = args[0]
            let arrB = args[1]
            let (arg1, arg2) =
              if matrixCount == 1 {
                (
                  transA
                    ? arrA.reshaped(inner, rows, stream: .device(device)).T
                    : arrA.reshaped(rows, inner, stream: .device(device)),
                  transB
                    ? arrB.reshaped(cols, inner, stream: .device(device)).T
                    : arrB.reshaped(inner, cols, stream: .device(device))
                )
              } else {
                (
                  transA
                    ? arrA.reshaped(matrixCount, inner, rows, stream: .device(device)).movedAxis(
                      source: 2, destination: 1, stream: .device(device))
                    : arrA.reshaped(matrixCount, rows, inner, stream: .device(device)),
                  transB
                    ? arrB.reshaped(matrixCount, cols, inner, stream: .device(device)).movedAxis(
                      source: 2, destination: 1, stream: .device(device))
                    : arrB.reshaped(matrixCount, inner, cols, stream: .device(device))
                )
              }
            return [
              MLX.matmul(arg1, arg2, stream: .device(device)).flattened(
                stream: .device(device))
            ]
          }
        if matmulCache[key] == nil {
          matmulCache[key] = fn
        }
        let result = fn([arrA, arrB])[0]
        MLX.asyncEval(result)
        return result
      }
    }
    return MLXData(backend: self, data: result)
  }

  internal struct AdamWKey: Equatable, Hashable {
    public let hasWeightDecay: Bool
    public let eps: Float
    public let count: Int
    public let dtype: Tensor.DType
  }

  internal var adamWCache = [AdamWKey: @Sendable ([MLXArray]) -> [MLXArray]]()

  override open func adamW(
    param: Tensor.Data,
    grad: Tensor.Data,
    moment1: Tensor.Data,
    moment2: Tensor.Data,
    beta1: Float,
    beta2: Float,
    eps: Float,
    weightDecay: Float,
    lr: Float,
    step: Float,
    count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> (param: Tensor.Data, moment1: Tensor.Data, moment2: Tensor.Data)
  {
    let paramArr = try await asArray(param, count, mlxDType(dtype))
    let gradArr = try await asArray(grad, count, mlxDType(dtype))
    let moment1Arr = try await asArray(moment1, count, mlxDType(dtype))
    let moment2Arr = try await asArray(moment2, count, mlxDType(dtype))

    let device = device
    let (newParam, newMoment1, newMoment2) = try await serialize {
      [self] () -> (MLXArray, MLXArray, MLXArray) in
      using(device: device) {
        let key = AdamWKey(
          hasWeightDecay: weightDecay != 0,
          eps: eps,
          count: count,
          dtype: dtype
        )
        let fn =
          adamWCache[key]
          ?? MLX.compile { (args: [MLXArray]) in
            let paramArr = args[0].asType(.float32)
            let gradArr = args[1].asType(.float32)
            let moment1Arr = args[2].asType(.float32)
            let moment2Arr = args[3].asType(.float32)
            let lr = args[4]
            let step = args[5]
            let beta1 = args[6]
            let beta2 = args[7]
            let weightDecay = args[8]

            var theta = !key.hasWeightDecay ? paramArr : (paramArr * (1 - lr * weightDecay))
            let newMoment1 = beta1 * moment1Arr + (1 - beta1) * gradArr
            let newMoment2 = beta2 * moment2Arr + (1 - beta2) * MLX.pow(gradArr, 2)
            let correctedM1 = newMoment1 / (1 - MLX.pow(beta1, step))
            let correctedM2 = newMoment2 / (1 - MLX.pow(beta2, step))
            theta = theta - lr * correctedM1 / (correctedM2.sqrt() + eps)
            return [
              theta.asType(args[0].dtype),
              newMoment1.asType(args[0].dtype),
              newMoment2.asType(args[0].dtype),
            ]
          }
        if adamWCache[key] == nil {
          adamWCache[key] = fn
        }
        let outs = fn([
          paramArr, gradArr, moment1Arr, moment2Arr, MLXArray(lr), MLXArray(step), MLXArray(beta1),
          MLXArray(beta2), MLXArray(weightDecay),
        ])
        return (outs[0], outs[1], outs[2])
      }
    }

    return (
      param: MLXData(backend: self, data: newParam),
      moment1: MLXData(backend: self, data: newMoment1),
      moment2: MLXData(backend: self, data: newMoment2)
    )
  }

  override open func defaultRandom() -> RandomGenerator {
    _defaultRandomMLX!
  }

  override open func createRandom() -> RandomGenerator {
    MLXRandomGenerator(mlxBackend: self, seed: Int.random(in: 0..<1_000_000_000))
  }

}

func stridesForShape(_ shape: [Int]) -> [Int] {
  var strides = [Int](repeating: 0, count: shape.count)
  for i in 0..<shape.count {
    strides[i] = shape[(i + 1)...].reduce(1, { $0 * $1 })
  }
  return strides
}
