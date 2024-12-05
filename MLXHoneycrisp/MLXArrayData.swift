import Accelerate
import Foundation
import HCBacktrace
import Honeycrisp
import MLX

extension MLXArray {
  func rawPointer() -> (ptr: UnsafeMutableRawPointer, isCopy: Bool, dealloc: () -> Void) {
    eval()
    let buf = asMTLBuffer(device: DummyMTLDevice(), noCopy: true) as! DummyMTLDevice.Buffer
    if let ptr = buf.pointer {
      alwaysAssert(buf.length == dtype.byteSize * shape.reduce(1, { $0 * $1 }))
      // In-place editing is possible
      return (
        ptr: ptr,
        isCopy: false,
        dealloc: { [self] in
          let _ = self  // Extend lifetime of array past usage of pointer.
        }
      )
    }

    let someData = asData().data
    let newPtr = UnsafeMutableRawPointer.allocate(byteCount: someData.count, alignment: 16)
    someData.withUnsafeBytes { b in
      newPtr.copyMemory(from: b.baseAddress!, byteCount: b.count)
    }
    return (ptr: newPtr, isCopy: true, dealloc: { newPtr.deallocate() })
  }
}

private class DummyMTLDevice: MTLDevice {
  func makeRenderPipelineState(
    tileDescriptor descriptor: MTLTileRenderPipelineDescriptor, options: MTLPipelineOption
  ) async throws -> (any MTLRenderPipelineState, MTLRenderPipelineReflection?) {
    fatalError()
  }

  func makeRenderPipelineState(
    tileDescriptor descriptor: MTLTileRenderPipelineDescriptor, options: MTLPipelineOption,
    reflection: AutoreleasingUnsafeMutablePointer<MTLAutoreleasedRenderPipelineReflection?>?
  ) throws -> any MTLRenderPipelineState {
    fatalError()
  }

  func makeComputePipelineState(
    descriptor: MTLComputePipelineDescriptor, options: MTLPipelineOption
  ) async throws -> (any MTLComputePipelineState, MTLComputePipelineReflection?) {
    fatalError()
  }

  func makeComputePipelineState(
    descriptor: MTLComputePipelineDescriptor, options: MTLPipelineOption,
    reflection: AutoreleasingUnsafeMutablePointer<MTLAutoreleasedComputePipelineReflection?>?
  ) throws -> any MTLComputePipelineState {
    fatalError()
  }

  func makeComputePipelineState(
    function computeFunction: any MTLFunction, options: MTLPipelineOption
  ) async throws -> (any MTLComputePipelineState, MTLComputePipelineReflection?) {
    fatalError()
  }

  func makeComputePipelineState(
    function computeFunction: any MTLFunction, options: MTLPipelineOption,
    reflection: AutoreleasingUnsafeMutablePointer<MTLAutoreleasedComputePipelineReflection?>?
  ) throws -> any MTLComputePipelineState {
    fatalError()
  }

  func makeComputePipelineState(function computeFunction: any MTLFunction) async throws
    -> any MTLComputePipelineState
  {
    fatalError()
  }

  func makeComputePipelineState(function computeFunction: any MTLFunction) throws
    -> any MTLComputePipelineState
  {
    fatalError()
  }

  func makeRenderPipelineState(
    descriptor: MTLMeshRenderPipelineDescriptor, options: MTLPipelineOption
  ) async throws -> (any MTLRenderPipelineState, MTLRenderPipelineReflection?) {
    fatalError()
  }

  func makeRenderPipelineState(descriptor: MTLRenderPipelineDescriptor, options: MTLPipelineOption)
    async throws -> (any MTLRenderPipelineState, MTLRenderPipelineReflection?)
  {
    fatalError()
  }

  func makeRenderPipelineState(
    descriptor: MTLRenderPipelineDescriptor, options: MTLPipelineOption,
    reflection: AutoreleasingUnsafeMutablePointer<MTLAutoreleasedRenderPipelineReflection?>?
  ) throws -> any MTLRenderPipelineState {
    fatalError()
  }

  /*func makeRenderPipelineState(descriptor: MTLRenderPipelineDescriptor, completionHandler: @escaping MTLNewRenderPipelineStateCompletionHandler) {
        fatalError()
    }*/

  func makeRenderPipelineState(descriptor: MTLRenderPipelineDescriptor) async throws
    -> any MTLRenderPipelineState
  {
    fatalError()
  }

  func makeRenderPipelineState(descriptor: MTLRenderPipelineDescriptor) throws
    -> any MTLRenderPipelineState
  {
    fatalError()
  }

  func makeLogState(descriptor: MTLLogStateDescriptor) throws -> any MTLLogState {
    fatalError()
  }

  func makeCommandQueue(descriptor: MTLCommandQueueDescriptor) -> (any MTLCommandQueue)? {
    fatalError()
  }

  func makeResidencySet(descriptor desc: MTLResidencySetDescriptor) throws -> any MTLResidencySet {
    fatalError()
  }

  func makeLibrary(source: String, options: MTLCompileOptions?) throws -> any MTLLibrary {
    fatalError()
  }

  func makeLibrary(
    source: String, options: MTLCompileOptions?,
    completionHandler: @escaping MTLNewLibraryCompletionHandler
  ) {
    fatalError()
  }

  func makeLibrary(data: dispatch_data_t) throws -> any MTLLibrary {
    fatalError()
  }

  func makeLibrary(URL url: URL) throws -> any MTLLibrary {
    fatalError()
  }

  func makeLibrary(filepath: String) throws -> any MTLLibrary {
    fatalError()
  }

  func makeLibrary(stitchedDescriptor descriptor: MTLStitchedLibraryDescriptor) throws
    -> any MTLLibrary
  {
    fatalError()
  }

  func makeLibrary(
    stitchedDescriptor descriptor: MTLStitchedLibraryDescriptor,
    completionHandler: @escaping MTLNewLibraryCompletionHandler
  ) {
    fatalError()
  }

  var name: String = ""

  var registryID: UInt64 = 0

  var architecture: MTLArchitecture { fatalError() }

  var maxThreadsPerThreadgroup: MTLSize { fatalError() }

  var isLowPower: Bool { fatalError() }

  var isHeadless: Bool { fatalError() }

  var isRemovable: Bool { fatalError() }

  var hasUnifiedMemory: Bool { fatalError() }

  var recommendedMaxWorkingSetSize: UInt64 { fatalError() }

  var location: MTLDeviceLocation { fatalError() }

  var locationNumber: Int { fatalError() }

  var maxTransferRate: UInt64 { fatalError() }

  var isDepth24Stencil8PixelFormatSupported: Bool { fatalError() }

  var readWriteTextureSupport: MTLReadWriteTextureTier { fatalError() }

  var argumentBuffersSupport: MTLArgumentBuffersTier { fatalError() }

  var areRasterOrderGroupsSupported: Bool { fatalError() }

  var supports32BitFloatFiltering: Bool { fatalError() }

  var supports32BitMSAA: Bool { fatalError() }

  var supportsQueryTextureLOD: Bool { fatalError() }

  var supportsBCTextureCompression: Bool { fatalError() }

  var supportsPullModelInterpolation: Bool { fatalError() }

  var areBarycentricCoordsSupported: Bool { fatalError() }

  var supportsShaderBarycentricCoordinates: Bool { fatalError() }

  var currentAllocatedSize: Int { fatalError() }

  func makeCommandQueue() -> (any MTLCommandQueue)? {
    fatalError()
  }

  func makeCommandQueue(maxCommandBufferCount: Int) -> (any MTLCommandQueue)? {
    fatalError()
  }

  func heapTextureSizeAndAlign(descriptor desc: MTLTextureDescriptor) -> MTLSizeAndAlign {
    fatalError()
  }

  func heapBufferSizeAndAlign(length: Int, options: MTLResourceOptions = []) -> MTLSizeAndAlign {
    fatalError()
  }

  func makeHeap(descriptor: MTLHeapDescriptor) -> (any MTLHeap)? {
    fatalError()
  }

  func makeBuffer(length: Int, options: MTLResourceOptions = []) -> (any MTLBuffer)? {
    fatalError()
  }

  func makeBuffer(
    bytes pointer: UnsafeRawPointer, length: Int, options: MTLResourceOptions = []
  ) -> (any MTLBuffer)? {
    Buffer(copying: pointer, length)
  }

  func makeBuffer(
    bytesNoCopy pointer: UnsafeMutableRawPointer, length: Int, options: MTLResourceOptions = [],
    deallocator: ((UnsafeMutableRawPointer, Int) -> Void)? = nil
  ) -> (any MTLBuffer)? {
    Buffer(pointer, length)
  }

  func makeDepthStencilState(descriptor: MTLDepthStencilDescriptor) -> (
    any MTLDepthStencilState
  )? {
    fatalError()
  }

  func makeTexture(descriptor: MTLTextureDescriptor) -> (any MTLTexture)? {
    fatalError()
  }

  func makeTexture(descriptor: MTLTextureDescriptor, iosurface: IOSurfaceRef, plane: Int) -> (
    any MTLTexture
  )? {
    fatalError()
  }

  func makeSharedTexture(descriptor: MTLTextureDescriptor) -> (any MTLTexture)? {
    fatalError()
  }

  func makeSharedTexture(handle sharedHandle: MTLSharedTextureHandle) -> (any MTLTexture)? {
    fatalError()
  }

  func makeSamplerState(descriptor: MTLSamplerDescriptor) -> (any MTLSamplerState)? {
    fatalError()
  }

  func makeDefaultLibrary() -> (any MTLLibrary)? {
    fatalError()
  }

  func makeDefaultLibrary(bundle: Bundle) throws -> any MTLLibrary {
    fatalError()
  }

  func makeFence() -> (any MTLFence)? {
    fatalError()
  }

  @available(*, deprecated, message: "Use supportsFamily() instead")
  func supportsFeatureSet(_ featureSet: MTLFeatureSet) -> Bool {
    fatalError()
  }

  func supportsFamily(_ gpuFamily: MTLGPUFamily) -> Bool {
    fatalError()
  }

  func supportsTextureSampleCount(_ sampleCount: Int) -> Bool {
    fatalError()
  }

  func minimumLinearTextureAlignment(for format: MTLPixelFormat) -> Int {
    fatalError()
  }

  func minimumTextureBufferAlignment(for format: MTLPixelFormat) -> Int {
    fatalError()
  }

  func __newRenderPipelineState(
    withMeshDescriptor descriptor: MTLMeshRenderPipelineDescriptor, options: MTLPipelineOption,
    reflection: AutoreleasingUnsafeMutablePointer<MTLAutoreleasedRenderPipelineReflection?>?
  ) throws -> any MTLRenderPipelineState {
    fatalError()
  }

  var maxThreadgroupMemoryLength: Int { fatalError() }

  var maxArgumentBufferSamplerCount: Int { fatalError() }

  var areProgrammableSamplePositionsSupported: Bool { fatalError() }

  func __getDefaultSamplePositions(
    _ positions: UnsafeMutablePointer<MTLSamplePosition>, count: Int
  ) {
    fatalError()
  }

  func makeArgumentEncoder(arguments: [MTLArgumentDescriptor]) -> (any MTLArgumentEncoder)? {
    fatalError()
  }

  func supportsRasterizationRateMap(layerCount: Int) -> Bool {
    fatalError()
  }

  func makeRasterizationRateMap(descriptor: MTLRasterizationRateMapDescriptor) -> (
    any MTLRasterizationRateMap
  )? {
    fatalError()
  }

  func makeIndirectCommandBuffer(
    descriptor: MTLIndirectCommandBufferDescriptor, maxCommandCount maxCount: Int,
    options: MTLResourceOptions = []
  ) -> (any MTLIndirectCommandBuffer)? {
    fatalError()
  }

  func makeEvent() -> (any MTLEvent)? {
    fatalError()
  }

  func makeSharedEvent() -> (any MTLSharedEvent)? {
    fatalError()
  }

  func makeSharedEvent(handle sharedEventHandle: MTLSharedEventHandle) -> (any MTLSharedEvent)? {
    fatalError()
  }

  var peerGroupID: UInt64 { fatalError() }

  var peerIndex: UInt32 { fatalError() }

  var peerCount: UInt32 { fatalError() }

  func makeIOHandle(url: URL) throws -> any MTLIOFileHandle {
    fatalError()
  }

  func makeIOCommandQueue(descriptor: MTLIOCommandQueueDescriptor) throws
    -> any MTLIOCommandQueue
  {
    fatalError()
  }

  func makeIOHandle(url: URL, compressionMethod: MTLIOCompressionMethod) throws
    -> any MTLIOFileHandle
  {
    fatalError()
  }

  func makeIOFileHandle(url: URL) throws -> any MTLIOFileHandle {
    fatalError()
  }

  func makeIOFileHandle(url: URL, compressionMethod: MTLIOCompressionMethod) throws
    -> any MTLIOFileHandle
  {
    fatalError()
  }

  func sparseTileSize(
    with textureType: MTLTextureType, pixelFormat: MTLPixelFormat, sampleCount: Int
  ) -> MTLSize {
    fatalError()
  }

  var sparseTileSizeInBytes: Int { fatalError() }

  func sparseTileSizeInBytes(sparsePageSize: MTLSparsePageSize) -> Int {
    fatalError()
  }

  func sparseTileSize(
    textureType: MTLTextureType, pixelFormat: MTLPixelFormat, sampleCount: Int,
    sparsePageSize: MTLSparsePageSize
  ) -> MTLSize {
    fatalError()
  }

  var maxBufferLength: Int { fatalError() }

  var counterSets: [any MTLCounterSet]? {
    get { fatalError() }
    set { fatalError() }
  }

  func makeCounterSampleBuffer(descriptor: MTLCounterSampleBufferDescriptor) throws
    -> any MTLCounterSampleBuffer
  {
    fatalError()
  }

  func __sampleTimestamps(
    _ cpuTimestamp: UnsafeMutablePointer<MTLTimestamp>,
    gpuTimestamp: UnsafeMutablePointer<MTLTimestamp>
  ) {
    fatalError()
  }

  func makeArgumentEncoder(bufferBinding: any MTLBufferBinding) -> any MTLArgumentEncoder {
    fatalError()
  }

  func supportsCounterSampling(_ samplingPoint: MTLCounterSamplingPoint) -> Bool {
    fatalError()
  }

  func supportsVertexAmplificationCount(_ count: Int) -> Bool {
    fatalError()
  }

  var supportsDynamicLibraries: Bool { fatalError() }

  var supportsRenderDynamicLibraries: Bool { fatalError() }

  func makeDynamicLibrary(library: any MTLLibrary) throws -> any MTLDynamicLibrary {
    fatalError()
  }

  func makeDynamicLibrary(url: URL) throws -> any MTLDynamicLibrary {
    fatalError()
  }

  func makeBinaryArchive(descriptor: MTLBinaryArchiveDescriptor) throws -> any MTLBinaryArchive {
    fatalError()
  }

  var supportsRaytracing: Bool { fatalError() }

  func accelerationStructureSizes(descriptor: MTLAccelerationStructureDescriptor)
    -> MTLAccelerationStructureSizes
  {
    fatalError()
  }

  func makeAccelerationStructure(size: Int) -> (any MTLAccelerationStructure)? {
    fatalError()
  }

  func makeAccelerationStructure(descriptor: MTLAccelerationStructureDescriptor) -> (
    any MTLAccelerationStructure
  )? {
    fatalError()
  }

  func heapAccelerationStructureSizeAndAlign(size: Int) -> MTLSizeAndAlign {
    fatalError()
  }

  func heapAccelerationStructureSizeAndAlign(descriptor: MTLAccelerationStructureDescriptor)
    -> MTLSizeAndAlign
  {
    fatalError()
  }

  var supportsFunctionPointers: Bool { fatalError() }

  var supportsFunctionPointersFromRender: Bool { fatalError() }

  var supportsRaytracingFromRender: Bool { fatalError() }

  var supportsPrimitiveMotionBlur: Bool { fatalError() }

  var shouldMaximizeConcurrentCompilation: Bool {
    get { fatalError() }
    set { fatalError() }
  }

  var maximumConcurrentCompilationTaskCount: Int { fatalError() }

  func isEqual(_ object: Any?) -> Bool {
    fatalError()
  }

  var hash: Int { fatalError() }

  var superclass: AnyClass? { fatalError() }

  func `self`() -> Self {
    fatalError()
  }

  func perform(_ aSelector: Selector!) -> Unmanaged<AnyObject>! {
    fatalError()
  }

  func perform(_ aSelector: Selector!, with object: Any!) -> Unmanaged<AnyObject>! {
    fatalError()
  }

  func perform(_ aSelector: Selector!, with object1: Any!, with object2: Any!) -> Unmanaged<
    AnyObject
  >! {
    fatalError()
  }

  func isProxy() -> Bool {
    fatalError()
  }

  func isKind(of aClass: AnyClass) -> Bool {
    fatalError()
  }

  func isMember(of aClass: AnyClass) -> Bool {
    fatalError()
  }

  func conforms(to aProtocol: Protocol) -> Bool {
    fatalError()
  }

  func responds(to aSelector: Selector!) -> Bool {
    fatalError()
  }

  var description: String { fatalError() }

  internal class Buffer: MTLBuffer {
    func contents() -> UnsafeMutableRawPointer {
      fatalError()
    }

    func __didModifyRange(_ range: NSRange) {
      fatalError()
    }

    func makeTexture(descriptor: MTLTextureDescriptor, offset: Int, bytesPerRow: Int) -> (
      any MTLTexture
    )? {
      fatalError()
    }

    func __addDebugMarker(_ marker: String, range: NSRange) {
      fatalError()
    }

    func removeAllDebugMarkers() {
      fatalError()
    }

    var remoteStorageBuffer: (any MTLBuffer)? { fatalError() }

    func makeRemoteBufferView(_ device: any MTLDevice) -> (any MTLBuffer)? {
      fatalError()
    }

    var gpuAddress: UInt64 { fatalError() }

    var label: String? {
      get { fatalError() }
      set { fatalError() }
    }

    var device: any MTLDevice { fatalError() }

    var cpuCacheMode: MTLCPUCacheMode { fatalError() }

    var storageMode: MTLStorageMode { fatalError() }

    var hazardTrackingMode: MTLHazardTrackingMode { fatalError() }

    var resourceOptions: MTLResourceOptions { fatalError() }

    func setPurgeableState(_ state: MTLPurgeableState) -> MTLPurgeableState {
      fatalError()
    }

    var heap: (any MTLHeap)? { fatalError() }

    var heapOffset: Int { fatalError() }

    var allocatedSize: Int { fatalError() }

    func makeAliasable() {
      fatalError()
    }

    func isAliasable() -> Bool {
      fatalError()
    }

    func __setOwnerWithIdentity(_ task_id_token: task_id_token_t) -> kern_return_t {
      fatalError()
    }

    func isEqual(_ object: Any?) -> Bool {
      fatalError()
    }

    var hash: Int { fatalError() }

    var superclass: AnyClass? { fatalError() }

    func `self`() -> Self {
      fatalError()
    }

    func perform(_ aSelector: Selector!) -> Unmanaged<AnyObject>! {
      fatalError()
    }

    func perform(_ aSelector: Selector!, with object: Any!) -> Unmanaged<AnyObject>! {
      fatalError()
    }

    func perform(_ aSelector: Selector!, with object1: Any!, with object2: Any!) -> Unmanaged<
      AnyObject
    >! {
      fatalError()
    }

    func isProxy() -> Bool {
      fatalError()
    }

    func isKind(of aClass: AnyClass) -> Bool {
      fatalError()
    }

    func isMember(of aClass: AnyClass) -> Bool {
      fatalError()
    }

    func conforms(to aProtocol: Protocol) -> Bool {
      fatalError()
    }

    func responds(to aSelector: Selector!) -> Bool {
      fatalError()
    }

    var description: String { fatalError() }

    public let pointer: UnsafeMutableRawPointer?
    public let length: Int
    public let isCopy: Bool

    public init(_ pointer: UnsafeRawPointer, _ length: Int) {
      self.pointer = UnsafeMutableRawPointer(mutating: pointer)
      self.length = length
      self.isCopy = false
    }

    public init(copying pointer: UnsafeRawPointer, _ length: Int) {
      self.pointer = nil
      self.length = length
      self.isCopy = true
    }
  }
}
