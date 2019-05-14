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

        #region Helper Functions

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

        private static int DftRealLength(params int[] n)
        {
            int length = 1;
            foreach (int dim in n)
                length *= dim;
            return length;
        }

        private static int DftComplexLength(params int[] n)
        {
            int length = 2;
            foreach (int dim in new ArraySegment<int>(n, 0, n.Length - 1))
                length *= dim;
            length *= n[n.Length - 1] / 2 + 1;
            return length;
        }

        private static int R2rLength(int n, FftwR2rKind kind)
        {
            switch (kind)
            {
                case FftwR2rKind.FFTW_R2HC:
                case FftwR2rKind.FFTW_HC2R:
                case FftwR2rKind.FFTW_DHT:
                    return n;
                case FftwR2rKind.FFTW_REDFT00:
                    return 2 * (n - 1);
                case FftwR2rKind.FFTW_REDFT10:
                case FftwR2rKind.FFTW_REDFT01:
                case FftwR2rKind.FFTW_REDFT11:
                    return 2 * n;
                case FftwR2rKind.FFTW_RODFT00:
                    return 2 * (n + 1);
                case FftwR2rKind.FFTW_RODFT10:
                case FftwR2rKind.FFTW_RODFT01:
                case FftwR2rKind.FFTW_RODFT11:
                    return 2 * n;
                default:
                    throw new ArgumentException("Real to real transform kind must be one of allowed values in " +
                        "FftwR2rKind enum.", nameof(kind));
            }
        }

        private static int R2rLength(int n0, int n1, FftwR2rKind kind0, FftwR2rKind kind1) =>
            R2rLength(n0, kind0) * R2rLength(n1, kind1);

        private static int R2rLength(int n0, int n1, int n2, FftwR2rKind kind0, FftwR2rKind kind1, FftwR2rKind kind2) =>
            R2rLength(n0, kind0) * R2rLength(n1, kind1) * R2rLength(n2, kind2);

        private static int R2rLength(int[] n, FftwR2rKind[] kind)
        {
            if (n.Length != kind.Length)
                throw new ArgumentException("Length of int[] n and FftwR2rKind[] kind must match.", nameof(kind));
            int length = 1;
            for (int i = 0; i < n.Length; i++)
                length *= R2rLength(n[i], kind[i]);
            return length;
        }

        private static bool IsInPlace(FftwArray inArray, FftwArray outArray) =>
            inArray.Pointer == outArray.Pointer;

        #endregion

        #region Basic Interface

        #region DFT

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

        #endregion

        #region R2C

        public static FftwPlan DftR2c1D(int n0, FftwArray inArray, FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int outLength = DftComplexLength(n0);
            int inLength = IsInPlace(inArray, outArray) ? outLength : DftRealLength(n0);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_r2c_1d(n0,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftR2c1D(int n0, out FftwArray inArray, out FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            inArray = new FftwArray(DftRealLength(n0));
            outArray = new FftwArray(DftComplexLength(n0));
            return DftR2c1D(n0, inArray, outArray, flags);
        }

        public static FftwPlan DftR2c1D(int n0, out FftwArray ioArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftComplexLength(n0));
            return DftR2c1D(n0, ioArray, ioArray, flags);
        }

        public static FftwPlan DftR2c2D(int n0, int n1, FftwArray inArray, FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int outLength = DftComplexLength(n0, n1);
            int inLength = IsInPlace(inArray, outArray) ? outLength : DftRealLength(n0, n1);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_r2c_2d(n0, n1,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftR2c2D(int n0, int n1, out FftwArray inArray, out FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            inArray = new FftwArray(DftRealLength(n0, n1));
            outArray = new FftwArray(DftComplexLength(n0, n1));
            return DftR2c2D(n0, n1, inArray, outArray, flags);
        }

        public static FftwPlan DftR2c2D(int n0, int n1, out FftwArray ioArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftComplexLength(n0, n1));
            return DftR2c2D(n0, n1, ioArray, ioArray, flags);
        }

        public static FftwPlan DftR2c3D(int n0, int n1, int n2, FftwArray inArray, FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int outLength = DftComplexLength(n0, n1, n2);
            int inLength = IsInPlace(inArray, outArray) ? outLength : DftRealLength(n0, n1, n2);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_r2c_3d(n0, n1, n2,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftR2c3D(int n0, int n1, int n2, out FftwArray inArray, out FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            inArray = new FftwArray(DftRealLength(n0, n1, n2));
            outArray = new FftwArray(DftComplexLength(n0, n1, n2));
            return DftR2c3D(n0, n1, n2, inArray, outArray, flags);
        }

        public static FftwPlan DftR2c3D(int n0, int n1, int n2, out FftwArray ioArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftComplexLength(n0, n1, n2));
            return DftR2c3D(n0, n1, n2, ioArray, ioArray, flags);
        }

        public static FftwPlan DftR2c(int[] n, FftwArray inArray, FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int outLength = DftComplexLength(n);
            int inLength = IsInPlace(inArray, outArray) ? outLength : DftRealLength(n);
            ValidateArray(inArray, outLength, nameof(inArray));
            ValidateArray(outArray, inLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_r2c(n.Length, n,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftR2c(int[] n, out FftwArray inArray, out FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            inArray = new FftwArray(DftRealLength(n));
            outArray = new FftwArray(DftComplexLength(n));
            return DftR2c(n, inArray, outArray, flags);
        }

        public static FftwPlan DftR2c(int[] n, out FftwArray ioArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftComplexLength(n));
            return DftR2c(n, ioArray, ioArray, flags);
        }

        #endregion

        #region C2R

        public static FftwPlan DftC2r1D(int n0, FftwArray inArray, FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int inLength = DftComplexLength(n0);
            int outLength = IsInPlace(inArray, outArray) ? inLength : DftRealLength(n0);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_c2r_1d(n0,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftC2r1D(int n0, out FftwArray inArray, out FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            inArray = new FftwArray(DftComplexLength(n0));
            outArray = new FftwArray(DftRealLength(n0));
            return DftC2r1D(n0, inArray, outArray, flags);
        }

        public static FftwPlan DftC2r1D(int n0, out FftwArray ioArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftComplexLength(n0));
            return DftC2r1D(n0, ioArray, ioArray, flags);
        }

        public static FftwPlan DftC2r2D(int n0, int n1, FftwArray inArray, FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int inLength = DftComplexLength(n0, n1);
            int outLength = IsInPlace(inArray, outArray) ? inLength : DftRealLength(n0, n1);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_c2r_2d(n0, n1,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftC2r2D(int n0, int n1, out FftwArray inArray, out FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            inArray = new FftwArray(DftComplexLength(n0, n1));
            outArray = new FftwArray(DftRealLength(n0, n1));
            return DftC2r2D(n0, n1, inArray, outArray, flags);
        }

        public static FftwPlan DftC2r2D(int n0, int n1, out FftwArray ioArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftComplexLength(n0, n1));
            return DftC2r2D(n0, n1, ioArray, ioArray, flags);
        }

        public static FftwPlan DftC2r3D(int n0, int n1, int n2, FftwArray inArray, FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int inLength = DftComplexLength(n0, n1, n2);
            int outLength = IsInPlace(inArray, outArray) ? inLength : DftRealLength(n0, n1, n2);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_c2r_3d(n0, n1, n2,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftC2r3D(int n0, int n1, int n2, out FftwArray inArray, out FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            inArray = new FftwArray(DftComplexLength(n0, n1, n2));
            outArray = new FftwArray(DftRealLength(n0, n1, n2));
            return DftC2r3D(n0, n1, n2, inArray, outArray, flags);
        }

        public static FftwPlan DftC2r3D(int n0, int n1, int n2, out FftwArray ioArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftComplexLength(n0, n1, n2));
            return DftC2r3D(n0, n1, n2, ioArray, ioArray, flags);
        }

        public static FftwPlan DftC2r(int[] n, FftwArray inArray, FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int inLength = DftComplexLength(n);
            int outLength = IsInPlace(inArray, outArray) ? inLength : DftRealLength(n);
            ValidateArray(inArray, outLength, nameof(inArray));
            ValidateArray(outArray, inLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_c2r(n.Length, n,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftC2r(int[] n, out FftwArray inArray, out FftwArray outArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            inArray = new FftwArray(DftComplexLength(n));
            outArray = new FftwArray(DftRealLength(n));
            return DftC2r(n, inArray, outArray, flags);
        }

        public static FftwPlan DftC2r(int[] n, out FftwArray ioArray,
            FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftComplexLength(n));
            return DftC2r(n, ioArray, ioArray, flags);
        }

        #endregion

        #region R2R

        public static FftwPlan DftR2r1D(int n0, FftwArray inArray, FftwArray outArray,
            FftwR2rKind kind0, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = R2rLength(n0, kind0);
            ValidateArray(inArray, length, nameof(inArray));
            ValidateArray(outArray, length, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_r2r_1d(n0,
                    inArray.Pointer, outArray.Pointer,
                    kind0, (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftR2r1D(int n0, out FftwArray inArray, out FftwArray outArray,
            FftwR2rKind kind0, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = R2rLength(n0, kind0);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return DftR2r1D(n0, inArray, outArray, kind0, flags);
        }

        public static FftwPlan DftR2r1D(int n0, out FftwArray ioArray,
            FftwR2rKind kind0, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(R2rLength(n0, kind0));
            return DftR2r1D(n0, ioArray, ioArray, kind0, flags);
        }

        public static FftwPlan DftR2r2D(int n0, int n1, FftwArray inArray, FftwArray outArray,
            FftwR2rKind kind0, FftwR2rKind kind1, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = R2rLength(n0, n1, kind0, kind1);
            ValidateArray(inArray, length, nameof(inArray));
            ValidateArray(outArray, length, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_r2r_2d(n0, n1,
                    inArray.Pointer, outArray.Pointer,
                    kind0, kind1, (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftR2r2D(int n0, int n1, out FftwArray inArray, out FftwArray outArray,
            FftwR2rKind kind0, FftwR2rKind kind1, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = R2rLength(n0, n1, kind0, kind1);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return DftR2r2D(n0, n1, inArray, outArray, kind0, kind1, flags);
        }

        public static FftwPlan DftR2r2D(int n0, int n1, out FftwArray ioArray,
            FftwR2rKind kind0, FftwR2rKind kind1, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(R2rLength(n0, n1, kind0, kind1));
            return DftR2r2D(n0, n1, ioArray, ioArray, kind0, kind1, flags);
        }

        public static FftwPlan DftR2r3D(int n0, int n1, int n2, FftwArray inArray, FftwArray outArray,
            FftwR2rKind kind0, FftwR2rKind kind1, FftwR2rKind kind2, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = R2rLength(n0, n1, n2, kind0, kind1, kind2);
            ValidateArray(inArray, length, nameof(inArray));
            ValidateArray(outArray, length, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_r2r_3d(n0, n1, n2,
                    inArray.Pointer, outArray.Pointer,
                    kind0, kind1, kind2, (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftR2r3D(int n0, int n1, int n2, out FftwArray inArray, out FftwArray outArray,
            FftwR2rKind kind0, FftwR2rKind kind1, FftwR2rKind kind2, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = R2rLength(n0, n1, n2, kind0, kind1, kind2);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return DftR2r3D(n0, n1, n2, inArray, outArray, kind0, kind1, kind2, flags);
        }

        public static FftwPlan DftR2r3D(int n0, int n1, int n2, out FftwArray ioArray,
            FftwR2rKind kind0, FftwR2rKind kind1, FftwR2rKind kind2, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(R2rLength(n0, n1, n2, kind0, kind1, kind2));
            return DftR2r3D(n0, n1, n2, ioArray, ioArray, kind0, kind1, kind2, flags);
        }

        public static FftwPlan DftR2r(int[] n, FftwArray inArray, FftwArray outArray,
            FftwR2rKind[] kind, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = R2rLength(n, kind);
            ValidateArray(inArray, length, nameof(inArray));
            ValidateArray(outArray, length, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_r2r(n.Length, n,
                    inArray.Pointer, outArray.Pointer,
                    kind, (uint_t)flags),
                inArray, outArray);
        }

        public static FftwPlan DftR2r(int[] n, out FftwArray inArray, out FftwArray outArray,
            FftwR2rKind[] kind, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = R2rLength(n, kind);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return DftR2r(n, inArray, outArray, kind, flags);
        }

        public static FftwPlan DftR2r(int[] n, out FftwArray ioArray,
            FftwR2rKind[] kind, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(R2rLength(n, kind));
            return DftR2r(n, ioArray, ioArray, kind, flags);
        }

        #endregion

        #endregion

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
