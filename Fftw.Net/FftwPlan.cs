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

        private FftwPlan(FftwFlags flags, IntPtr pointer, params FftwArray[] arrays)
        {
            if (pointer == IntPtr.Zero)
                throw new ArgumentException("A NULL plan was returned by the FFTW planner method. " +
                    "This is probably due to a bad flag. Please check the FFTW documentation for details.");
            this.pointer = pointer;
            mustAlign = (flags & FftwFlags.FFTW_UNALIGNED) == 0;
            foreach (var array in arrays)
                mustAlign = mustAlign && array.FftwAllocated;
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

        private static int DftLength(params int[] n)
        {
            int length = 2;
            foreach (int dim in n)
                length *= dim;
            return length;
        }

        public static FftwPlan Dft1D(int n0, FftwArray inArray, FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n0);
            ValidateSign(sign);
            ValidateArray(inArray, length, nameof(inArray));
            ValidateArray(outArray, length, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_1d(n0,
                    inArray.Pointer, outArray.Pointer,
                    (int_t)sign, (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan Dft1D(int n0, out FftwArray inArray, out FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n0);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return Dft1D(n0, inArray, outArray, sign, flags);
        }

        public static FftwPlan Dft1D(int n0, out FftwArray ioArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftLength(n0));
            return Dft1D(n0, ioArray, ioArray, sign, flags);
        }

        public static FftwPlan Dft2D(int n0, int n1, FftwArray inArray, FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n0, n1);
            ValidateSign(sign);
            ValidateArray(inArray, length, nameof(inArray));
            ValidateArray(outArray, length, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_2d(n0, n1,
                    inArray.Pointer, outArray.Pointer,
                    (int_t)sign, (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan Dft2D(int n0, int n1, out FftwArray inArray, out FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n0, n1);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return Dft2D(n0, n1, inArray, outArray, sign, flags);
        }

        public static FftwPlan Dft2D(int n0, int n1, out FftwArray ioArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftLength(n0, n1));
            return Dft2D(n0, n1, ioArray, ioArray, sign, flags);
        }

        public static FftwPlan Dft3D(int n0, int n1, int n2, FftwArray inArray, FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n0, n1, n2);
            ValidateSign(sign);
            ValidateArray(inArray, length, nameof(inArray));
            ValidateArray(outArray, length, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_3d(n0, n1, n2,
                    inArray.Pointer, outArray.Pointer,
                    (int_t)sign, (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan Dft3D(int n0, int n1, int n2, out FftwArray inArray, out FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n0, n1, n2);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return Dft3D(n0, n1, n2, inArray, outArray, sign, flags);
        }

        public static FftwPlan Dft3D(int n0, int n1, int n2, out FftwArray ioArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftLength(n0, n1, n2));
            return Dft3D(n0, n1, n2, ioArray, ioArray, sign, flags);
        }

        public static FftwPlan Dft(int[] n, FftwArray inArray, FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n);
            ValidateSign(sign);
            ValidateArray(inArray, length, nameof(inArray));
            ValidateArray(outArray, length, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft(n.Length, n,
                    inArray.Pointer, outArray.Pointer,
                    (int_t)sign, (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan Dft(int[] n, out FftwArray inArray, out FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return Dft(n, inArray, outArray, sign, flags);
        }

        public static FftwPlan Dft(int[] n, out FftwArray ioArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftLength(n));
            return Dft(n, ioArray, ioArray, sign, flags);
        }

        private static int R2cRealLength(bool inPlace, params int[] n)
        {
            int length = 2;
            foreach (int dim in n)
                length *= dim;
            return length;
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
