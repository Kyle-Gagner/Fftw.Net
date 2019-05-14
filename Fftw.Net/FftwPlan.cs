using System;

using static Fftw.Net.FftwBindings;

using int_t = System.Int32;
using uint_t = System.UInt32;

namespace Fftw.Net
{
    public class FftwPlan : IDisposable
    {
        private IntPtr pointer;

        private bool mustAlign;

        /// <summary>
        /// Set true on disposal to protect from accidentally trying to free resources multiple times
        /// </summary>
        private bool disposed = false;

        /// <summary>
        /// An object used as a lock to ensure thread safety during object disposal
        /// </summary>
        private object lockObject = new object();

        private FftwPlan(IntPtr pointer)
        {
            this.pointer = pointer;
        }

        private FftwPlan(FftwArray inArray, int inLength, FftwArray outArray, int outLength,
            FftwFlags flags, IntPtr pointer) : this(pointer)
        {
            mustAlign = inArray.FftwAllocated && outArray.FftwAllocated && (flags & FftwFlags.FFTW_UNALIGNED) == 0;
        }

        private static void ValidateSign(FftwSign sign)
        {
            if (sign != FftwSign.FFTW_FORWARD && sign != FftwSign.FFTW_BACKWARD)
                throw new ArgumentException($"Sign is not one of FFTW_FORWARD, FFTW_BACKWARD", nameof(sign));
        }

        private static void ValidateArray(FftwArray array, int length, string argument)
        {
            if (array.Length < length)
                throw new ArgumentException($"Array length must be at least {length}.", argument);
        }

        public static FftwPlan Dft1D(int n0, FftwArray inArray, FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = 2 * n0;
            ValidateSign(sign);
            ValidateArray(inArray, length, nameof(inArray));
            ValidateArray(outArray, length, nameof(outArray));
            return new FftwPlan(inArray, length, outArray, length, flags,
                fftw_plan_dft_1d(n0,
                    inArray.Pointer, outArray.Pointer,
                    (int_t)sign, (uint_t)flags));
        }

        public static FftwPlan Dft1D(int n0, out FftwArray inArray, out FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            inArray = new FftwArray(2 * n0);
            outArray = new FftwArray(2 * n0);
            return Dft1D(n0, inArray, outArray, sign, flags);
        }

        public static FftwPlan Dft1D(int n0, out FftwArray ioArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(2 * n0);
            return Dft1D(n0, ioArray, ioArray, sign, flags);
        }

        public void Execute()
        {
            fftw_execute(pointer);
        }

        public void Dispose()
        {
            lock (lockObject)
            {
                if (disposed)
                    return;
                disposed = true;
            }
            fftw_destroy_plan(pointer);
            GC.SuppressFinalize(this);
        }

        ~FftwPlan()
        {
            Dispose();
        }
    }
}
