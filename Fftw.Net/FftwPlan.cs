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

        public FftwArray In { get; }

        public FftwArray Out { get; }

        private FftwPlan(IntPtr pointer)
        {
            
            this.pointer = pointer;
        }

        private FftwPlan(FftwArray inArray, int inLength, FftwArray outArray, int outLength,
            FftwFlags flags, IntPtr pointer) : this(pointer)
        {
            mustAlign = inArray.FftwAllocated && outArray.FftwAllocated && (flags & FftwFlags.FFTW_UNALIGNED) == 0;
            In = inArray;
            Out = outArray;
        }

        private FftwPlan(FftwArray ioArray, int ioLength,
            FftwFlags flags, IntPtr pointer) : this(pointer)
        {
            mustAlign = ioArray.FftwAllocated && (flags & FftwFlags.FFTW_UNALIGNED) == 0;
            In = ioArray;
            Out = ioArray;
        }

        private static void ValidateSign(FftwSign sign)
        {
            if (sign != FftwSign.FFTW_FORWARD && sign != FftwSign.FFTW_BACKWARD)
                throw new ArgumentException($"Sign is not one of FFTW_FORWARD, FFTW_BACKWARD", nameof(sign));
        }

        private static void ValidateArray(ref FftwArray array, int length, string argument)
        {
            if (array == null)
                array = new FftwArray(length);
            else if (array.Length < length)
                throw new ArgumentException($"Array length must be at least {length}.", argument);
        }

        public static FftwPlan Dft(int n0, FftwArray inArray, FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = 2 * n0;
            ValidateSign(sign);
            ValidateArray(ref inArray, length, nameof(inArray));
            ValidateArray(ref outArray, length, nameof(outArray));
            return new FftwPlan(inArray, length, outArray, length, flags,
                fftw_plan_dft_1d(n0,
                    inArray.Pointer, outArray.Pointer,
                    (int_t)sign, (uint_t)flags));
        }

        public static FftwPlan Dft(int n0, FftwArray ioArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = 2 * n0;
            ValidateSign(sign);
            ValidateArray(ref ioArray, length, nameof(ioArray));
            return new FftwPlan(ioArray, length, flags,
                fftw_plan_dft_1d(n0,
                    ioArray.Pointer, ioArray.Pointer,
                    (int_t)sign, (uint_t)flags));
        }
        
        public static FftwPlan Dft(int n0,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE) =>
            Dft(n0, null, null, sign, flags);

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
