using System;
using System.Runtime.InteropServices;

using static Fftw.Net.FftwBindings;

namespace Fftw.Net
{
    public class FftwArray: IDisposable
    {
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
        /// An object used as a lock to ensure thread safety during object disposal
        /// </summary>
        private object lockObject = new object();

        /// <summary>
        /// A pointer to the contents of the array, either supplied externally, allocated on the unmanaged heap, or
        /// from a pinned array
        /// </summary>
        public IntPtr Pointer { get; }

        /// <summary>
        /// The length of the array
        /// </summary>
        public int Length { get; }

        /// <summary>
        /// For the purpose of ensuring that plans which expect aligned data recieve it for new array executes, this
        /// property may be used to determine whether the plan's arrays are expected to be aligned.
        /// </summary>
        internal bool FftwAllocated => owning;

        public Span<double> Span
        {
            get
            {
                unsafe
                {
                    return new Span<double>(Pointer.ToPointer(), Length);
                }
            }
        }

        public FftwArray(int length)
        {
            Length = length;
            Pointer = fftw_malloc((UIntPtr)(sizeof(double) * Length));
            owning = true;
        }

        public FftwArray(double[] array)
        {
            Length = array.Length;
            handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            Pointer = handle.Value.AddrOfPinnedObject();
        }

        public unsafe FftwArray(IntPtr pointer, int length)
        {
            Length = length;
            Pointer = pointer;
        }

        protected virtual void Dispose(bool disposing)
        {
            lock (lockObject)
            {
                if (disposed)
                    return;
                disposed = true;
            }
            if (owning)
                fftw_free(Pointer);
            else if (handle.HasValue)
                handle.Value.Free();
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~FftwArray()
        {
            Dispose(false);
        }
    }
}
