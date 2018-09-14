using System;
using static Fftw.Net.FftwBindings;

namespace Fftw.Net
{
    public class FftwArray : IDisposable
    {
        private bool disposed = false;
        private IntPtr pointer;
        private long length;
        private long size;
        private bool fftw_owns;

        public Span<double> Span
        {
            get
            {
                unsafe
                {
                    return new Span<double>(pointer.ToPointer(), (int)size);
                }
            }
        }

        public long Length => length;

        public FftwArray(long length)
        {
            this.length = length;
            this.size = sizeof(double) * length;
            unsafe
            {
                pointer = fftw_malloc((UIntPtr)size);
            }
            this.fftw_owns = true;
        }

        unsafe public FftwArray(double* pointer, long length)
        {
            this.length = length;
            this.size = sizeof(double) * length;
            this.pointer = new IntPtr(pointer);
            this.fftw_owns = false;
        }

        public FftwArray Slice(long start)
        {
            if (start > length)
                throw new ArgumentOutOfRangeException("start");
            unsafe
            {
                return new FftwArray((double*)pointer + start, length - start);
            }
        }

        public FftwArray Slice(long start, long length)
        {
            if (start > this.length)
                throw new ArgumentOutOfRangeException("start");
            if (length > this.length - start)
                throw new ArgumentOutOfRangeException("length");
            unsafe
            {
                return new FftwArray((double*)pointer + start, length - start);
            }
        }

        public static explicit operator IntPtr(FftwArray array) => array.pointer;

        public unsafe static explicit operator double*(FftwArray array) => (double*)array.pointer.ToPointer();

        private void Dispose(bool disposing)
        {
            unsafe
            {
                fftw_free(pointer);
            }
        }

        public void Dispose()
        {
            if (fftw_owns)
            {
                lock (this)
                {
                    if (!disposed)
                    {
                        disposed = true;
                        Dispose(true);
                        GC.SuppressFinalize(this);
                    }
                }
            }
            else
            {
                GC.SuppressFinalize(this);
            }
        }

        ~FftwArray()
        {
            if (fftw_owns)
            {
                if (!disposed)
                {
                    disposed = true;
                    Dispose(false);
                }
            }
        }
    }
}
