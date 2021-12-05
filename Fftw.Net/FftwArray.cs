using System;
using System.Runtime.InteropServices;
using System.Numerics;
using System.Threading;

using static Fftw.Net.FftwBindings;

namespace Fftw.Net
{
    /// <summary>
    /// FftwArray encapsulates a span of underlying memory and manages the lifetime of these underlying memory
    /// resources.
    /// </summary>
    /// <remarks>
    /// FftwArray supports arrays backed by unmanaged memory allocations from FFTW (preferred), pinned managed
    /// array objects (long lived pins may impact garbage collector performance), and arbitrary memory supplied by a
    /// pointer (unsafe, use with care). SIMD is supported when using arrays backed by unmanaged allocations from FFTW
    /// and the native library was compiled with SIMD support. FftwArray has limited thread safety. Slicing and property
    /// getters are thread safe. Properties and methods are not checked for disposal state.
    /// </remarks>
    public class FftwArray: IDisposable
    {
        /// <summary>
        /// The number of yet to be disposed slices of the array, applicable for roots of slices
        /// </summary>
        private int slices = 0;

        /// <summary>
        /// The number of yet to be disposed plans in which the array is involved
        /// </summary>
        private int plans = 0;

        /// <summary>
        /// The root array from which the slice was created, applicable for slices
        /// <summary>
        private FftwArray root = null;

        /// <summary>
        /// The length of the array
        /// <summary>
        private long length;

        /// <summary>
        /// Indicates whether this object owns memory at Pointer which has been allocated through FFTW
        /// </summary>
        private bool owning = false;

        /// <summary>
        /// This handle will have a value if Pointer refers to a pinned object
        /// </summary>
        private GCHandle? handle = null;
        
        /// <summary>
        /// Set true on disposal to protect from accidentally trying to free resources multiple times
        /// </summary>
        private bool disposed = false;

        /// <summary>
        /// A pointer to the contents of the array, either supplied externally, allocated on the unmanaged heap, or
        /// from a pinned array
        /// </summary>
        public IntPtr Pointer { get; }

        /// <summary>
        /// Gets the length of the array.
        /// </summary>
        /// <exception cref="OverflowException">The length exceeds <see cref="Int32.MaxValue"/>.</exception>
        public int Length => checked((int)length);

        /// <summary>
        /// Gets the length of the array.
        /// </summary>
        public long LongLength => length;

        /// <summary>
        /// For the purpose of ensuring that plans which expect aligned data recieve it for new array executes, this
        /// property may be used to determine whether the plan's arrays are expected to be aligned.
        /// </summary>
        internal bool FftwAllocated => owning;

        /// <summary>
        /// Gets a <see cref="Span<T>"/> for the underlying memory backing the buffer as double values
        /// </summary>
        /// <remarks>
        /// Due to its safety and efficiency, <see cref="Span<T>"/> is the preferred means of accessing the memory.
        /// Spans are limited by the size of a 32-bit integer. To access the span on very large arrays,
        /// use <see cref="Slice(long,long)"/> to create a smaller array backed by the same memory, then use the span on
        /// that smaller array.
        /// </remarks>
        /// <exception cref="OverflowException">The length exceeds <see cref="Int32.MaxValue"/>.</exception>
        public Span<double> DoubleSpan
        {
            get
            {
                unsafe
                {
                    try
                    {
                        return new Span<double>(Pointer.ToPointer(), Length);
                    }
                    catch (OverflowException ex)
                    {
                        throw new OverflowException("The length of the array exceeds Int32.MaxValue." +
                            "Try slicing the array to a reasonable size before getting the Span on the slice.", ex);
                    }
                }
            }
        }

        /// <summary>
        /// Gets a <see cref="Span<T>"/> for the underlying memory backing the buffer as complex values
        /// </summary>
        /// <remarks>
        /// Due to its safety and efficiency, <see cref="Span<T>"/> is the preferred means of accessing the memory.
        /// Spans are limited by the size of a 32-bit integer. To access the span on very large arrays,
        /// use <see cref="Slice(long,long)"/> to create a smaller array backed by the same memory, then use the span on
        /// that smaller array.
        /// </remarks>
        /// <exception cref="OverflowException">The length exceeds <see cref="Int32.MaxValue"/>.</exception>
        public Span<Complex> ComplexSpan
        {
            get
            {
                unsafe
                {
                    try
                    {
                        return new Span<Complex>(Pointer.ToPointer(), Length / 2);
                    }
                    catch (OverflowException ex)
                    {
                        throw new OverflowException("The length of the array exceeds Int32.MaxValue." +
                            "Try slicing the array to a reasonable size before getting the Span on the slice.", ex);
                    }
                }
            }
        }

        /// <summary>
        /// Initializes an instance of <see cref="FftwArray"/> backed by memory allocated on the unmanaged heap using
        /// fftw_malloc for the lifetime of the array until it is disposed at which point it is freed with fftw_free
        /// <summary/>
        /// <param name="length">The number of elements in the array</param>
        /// <exception cref="ArgumentOutOfRangeException">Specified length is less than 1.</exception>
        /// <exception cref="OverflowException">
        /// The calculation of the physical allocation size overflowed or the result could not be cast to
        /// <see cref="UIntPtr"/>.
        /// </exception>
        /// <exception cref="OutOfMemoryException">The physical allocation size could not be satisfied.</exception>
        public FftwArray(long length)
        {
            if (length < 1)
                throw new ArgumentException("Arrays must contain at least one element.", nameof(length));
            this.length = length;
            try
            {
                checked
                {
                    Pointer = fftw_malloc((UIntPtr)(sizeof(double) * length));
                }
            }
            catch (OverflowException ex)
            {
                throw new OverflowException("The physical size of the allocation is unreasonably large.", ex);
            }
            if (Pointer == IntPtr.Zero)
                throw new OutOfMemoryException("The system could not allocate the requested amount of memory.");
            owning = true;
        }

        /// <summary>
        /// Initializes an instance of <see cref="FftwArray"/> backed by memory allocated on the unmanaged heap using
        /// fftw_malloc for the lifetime of the array until it is disposed at which point it is freed with fftw_free
        /// <summary/>
        /// <param name="length">The number of elements in the array</param>
        /// <exception cref="ArgumentOutOfRangeException">Specified length is less than 1.</exception>
        /// <exception cref="OverflowException">
        /// The calculation of the physical allocation size overflowed or the result could not be cast to
        /// <see cref="UIntPtr"/>.
        /// </exception>
        /// <exception cref="OutOfMemoryException">The physical allocation size could not be satisfied.</exception>
        public FftwArray(int length) : this((long)length) { }

        /// <summary>
        /// Initializes an instance of <see cref="FftwArray"/> backed by an array object which is pinned for the
        /// lifetime of the array until it is disposed. Modifying data in the FftwArray alters the data in the
        /// underlying Array object.
        /// <summary>
        /// <remarks>
        /// Pinning large objects can severely affect the efficiency of the runtime's garbage collector. Objects
        /// constructed by this method should be disposed as soon as possible.
        /// </remarks>
        public FftwArray(double[] array)
        {
            length = array.LongLength;
            handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            Pointer = handle.Value.AddrOfPinnedObject();
        }

        /// <summary>
        /// Initializes an instance of <see cref="FftwArray"/> backed by arbitrary memory. No action is taken upon
        /// disposal.
        /// </summary>
        /// <remarks>
        /// Throws an <see cref="ArgumentOutOfRangeException"/> for length &lt; 1.
        /// </remarks>
        /// <param name="pointer">A pointer to the memory used by the array</param>
        /// <param name="length">The number of elements in the array</param>
        public unsafe FftwArray(IntPtr pointer, long length)
        {
            if (length < 1)
                throw new ArgumentOutOfRangeException(nameof(length));
            this.length = length;
            Pointer = pointer;
        }

        /// <summary>
        /// Initializes an instance of <see cref="FftwArray"/> backed by arbitrary memory. No action is taken upon
        /// disposal.
        /// </summary>
        /// <param name="pointer">A pointer to the memory used by the array</param>
        /// <param name="length">The number of elements in the array</param>
        /// <exception cref="ArgumentOutOfRangeException">Specified length is less than 1.</exception>
        public unsafe FftwArray(IntPtr pointer, int length) : this(pointer, (long)length) { }

        /// <summary>
        /// Initializes an instance of <see cref="FftwArray"/> backed by memory owned by a root array. Instantiation
        /// increment's the root's slices counter and disposal decrements it again.
        /// </summary>
        /// <param name="pointer">A pointer to the memory used by the array</param>
        /// <param name="length">The number of elements in the array</param>
        /// <param name="root">The root array which owns the memory</param>
        private FftwArray(IntPtr pointer, long length, FftwArray root)
        {
            Pointer = pointer;
            this.length = length;
            this.root = root;
            root.IncrementSlices();
        }

        /// <summary>
        /// Slices a subsequence from an array starting at a specified index and running for a specified length.
        /// </summary>
        /// <remarks>
        /// The slice is backed by the same memory and must be disposed before the root from which it was sliced may be
        /// disposed. A slice may be sliced, but both the slice and its slice share the same root. Slices of a root
        /// may be disposed in any order. Slices are useful for accessing the data in large arrays or for advanced
        /// use-cases where passing different portions of a single array as arguments to FFTW routines is desirable.
        /// </remarks>
        /// <param name="start">The index of the first element of the slice</param>
        /// <param name="length">The length of the slice</param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// The specified start and length are not a subset of the array or length is less than 1.
        /// </exception>
        /// <exception cref="OverflowException">Arithmetic overflowed computing the bounds or slice pointer.</exception>
        public FftwArray Slice(long start, long length)
        {
            try
            {
                checked
                {
                    if (start < 0 || start >= this.length)
                        throw new ArgumentOutOfRangeException(nameof(start));
                    if (length < 1 || start + length > this.length)
                        throw new ArgumentOutOfRangeException(nameof(length));
                    unsafe
                    {
                        return new FftwArray(new IntPtr((double*)Pointer.ToPointer() + start), length, root ?? this);
                    }
                }
            }
            catch (OverflowException ex)
            {
                throw new OverflowException("Arithmetic overflow in computing the slice.", ex);
            }
        }

        /// <summary>
        /// Slices a subsequence from an array starting at a specified index and running for a specified length.
        /// </summary>
        /// <remarks>
        /// The slice is backed by the same memory and must be disposed before the root from which it was sliced may be
        /// disposed. A slice may be sliced, but both the slice and its slice share the same root. Slices of a root
        /// may be disposed in any order. Slices are useful for accessing the data in large arrays or for advanced
        /// use-cases where passing different portions of a single array as arguments to FFTW routines is desirable.
        /// </remarks>
        /// <param name="start">The index of the first element of the slice</param>
        /// <param name="length">The length of the slice</param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// The specified start and length are not a subset of the array or length is less than 1.
        /// </exception>
        /// <exception cref="OverflowException">Arithmetic overflowed computing the bounds or slice pointer.</exception>
        public FftwArray Slice(int start, int length) => Slice((int)start, (int)length);

        /// <summary>
        /// Slices a subsequence from an array starting at a specified index and running to the end of the array.
        /// </summary>
        /// <remarks>
        /// The slice is backed by the same memory and must be disposed before the root from which it was sliced may be
        /// disposed. A slice may be sliced, but both the slice and its slice share the same root. Slices of a root
        /// may be disposed in any order. Slices are useful for accessing the data in large arrays or for advanced
        /// use-cases where passing different portions of a single array as arguments to FFTW routines is desirable.
        /// </remarks>
        /// <param name="start">The index of the first element of the slice</param>
        /// <exception cref="ArgumentOutOfRangeException">The specified start is outside the array.</exception>
        /// <exception cref="OverflowException">Arithmetic overflowed computing the bounds or slice pointer.</exception>
        public FftwArray Slice(long start) => Slice(start, length - start);

        /// <summary>
        /// Slices a subsequence from an array starting at a specified index and running to the end of the array.
        /// </summary>
        /// <remarks>
        /// The slice is backed by the same memory and must be disposed before the root from which it was sliced may be
        /// disposed. A slice may be sliced, but both the slice and its slice share the same root. Slices of a root
        /// may be disposed in any order. Slices are useful for accessing the data in large arrays or for advanced
        /// use-cases where passing different portions of a single array as arguments to FFTW routines is desirable.
        /// </remarks>
        /// <param name="start">The index of the first element of the slice</param>
        /// <exception cref="ArgumentOutOfRangeException">The specified start is outside the array.</exception>
        /// <exception cref="OverflowException">Arithmetic overflowed computing the bounds or slice pointer.</exception>
        public FftwArray Slice(int start) => Slice((int)start);

        /// <summary>
        /// Atomically increments the count of yet to be disposed plans holding a reference to the array.
        /// </summary>
        internal void IncrementPlans() => Interlocked.Increment(ref plans);

        /// <summary>
        /// Atomically decrements the count of yet to be disposed plans holding a reference to the array.
        /// </summary>
        internal void DecrementPlans() => Interlocked.Decrement(ref plans);

        /// <summary>
        /// Atomically increments the count of yet to be disposed slices holding a reference to the array.
        /// </summary>
        private void IncrementSlices() => Interlocked.Increment(ref slices);

        /// <summary>
        /// Atomically decrements the count of yet to be disposed slices holding a reference to the array.
        /// </summary>
        private void DecrementSlices() => Interlocked.Decrement(ref slices);

        /// <summary>
        /// Implementation of Dispose pattern.
        /// </summary>
        /// <param name="disposing">True if disposing, false if finalizing.</param>
        protected virtual void Dispose(bool disposing)
        {
            // protect from multiple disposal
            if (disposed) return;
            if (disposing)
            {
                if (root == null && slices != 0 || plans != 0)
                {
                    // In this event, user disposal was attempted while other objects are potentially still using the
                    // array's memory. This is a very bad situation, but it is inappropriate to throw an exception since
                    // exceptions during disposal are generally handled quite poorly. Instead, we'll keep disposed false
                    // and let the finalizer run. The resources may not be cleaned up quickly, but it will happen
                    // eventually.
                    return;
                }
                GC.SuppressFinalize(this);
                if (root != null)
                    root.DecrementSlices();
                root = null;
            }
            if (owning)
                fftw_free(Pointer);
            else if (handle.HasValue)
                handle.Value.Free();
            disposed = true;
        }

        /// </inheritdoc>
        public void Dispose() => Dispose(true);

        ~FftwArray() => Dispose(false);
    }
}
