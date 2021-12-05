using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

using static Fftw.Net.FftwBindings;

using int_t = System.Int32;
using uint_t = System.UInt32;

namespace Fftw.Net
{
    /// <summary>
    /// FftwPlan is a base class for encapsulations of FFTW plan objects and contains the static factory methods for
    /// creating instances of plan objects.
    /// </summary>
    public class FftwPlan : IDisposable
    {
        #region Private Nested Types

        /// <summary>
        /// Uniquely identifies a new-array execute function.
        /// </summary>
        private enum NewArrayExecFn
        {
            Dft,
            SplitDft,
            DftR2c,
            SplitDftR2c,
            DftC2r,
            SplitDftC2r,
            R2r
        }

        /// <summary>
        /// Contains information on the constraints imposed on new-array execute methods including the appropriate
        /// new-array execute function, minimum lengths for the arrays, and whether the plan is in-place.
        /// </summary>
        private struct NewArrayExecConstraints
        {
            public NewArrayExecFn newArrayExecFn;
            public long length0, length1, length2, length3;
            public bool inPlace;

            public NewArrayExecConstraints(NewArrayExecFn newArrayExecFn, bool inPlace,
                long length0, long length1, long length2 = 0, long length3 = 0)
            {
                this.newArrayExecFn = newArrayExecFn;
                this.inPlace = inPlace;
                this.length0 = length0;
                this.length1 = length1;
                this.length2 = length2;
                this.length3 = length3;
            }
        }

        #endregion

        #region Private Fields

        private IntPtr pointer { get; }

        private bool mustAlign;

        /// <summary>
        /// Set true on disposal to protect from accidentally trying to free resources multiple times
        /// </summary>
        private bool disposed = false;

        /// <summary>
        /// The FftwArray objects used by the plan
        /// </summary>
        private FftwArray[] arrays;

        /// <summary>
        /// Whether <see cref="arrays"/> are owned by the plan (constructed by the plan factory method)
        /// </summary>
        private bool owned;
        
        /// <summary>
        /// Constraints on the new-array execute methods
        /// </summary>
        private NewArrayExecConstraints constraints;

        #endregion

        #region Private Constructor

        /// <summary>
        /// Initializes an instance of <see cref="FftwPlan"/>.
        /// </summary>
        /// <param name="flags">The flags passed to the planner routine.</param>
        /// <param name="pointer">The pointer to the plan returned by the planner routine</param>
        /// <param name="newArrayExecConstraints">The new-array execute constraints for this plan</param>
        /// <param name="arrays">The arrays used by the planner routine</param>
        private FftwPlan(FftwFlags flags, IntPtr pointer, NewArrayExecConstraints constraints, params FftwArray[] arrays)
        {
            if (pointer == IntPtr.Zero)
                throw new ArgumentException("A NULL plan was returned by the FFTW planner method. " +
                    "This is probably due to a bad flag. Please check the FFTW documentation for details.");
            this.pointer = pointer;
            this.constraints = constraints;
            this.arrays = arrays;
            mustAlign = (flags & FftwFlags.FFTW_UNALIGNED) == 0;
            foreach (var array in arrays)
                array.IncrementPlans();
            // bit of a hack - if a planning method calls another planning method, it is because the first is a wrapper
            // these wrappers allocate the arrays for the user, and so these arrays are owned by the plan
            var trace = new StackTrace();
            if (trace.FrameCount > 2 && trace.GetFrame(2).GetMethod().DeclaringType == typeof(FftwPlan))
                owned = true;
        }

        #endregion

        #region Helper Functions

        /// <summary>
        /// Validates the sign argument.
        /// </summary>
        /// <param name="sign">The sign argument</param>
        /// <exception cref="ArgumentException">Validation check failed</exception>
        private static void ValidateSign(FftwSign sign)
        {
            if (sign != FftwSign.FFTW_FORWARD && sign != FftwSign.FFTW_BACKWARD)
                throw new ArgumentException($"Sign is not one of FFTW_FORWARD, FFTW_BACKWARD", nameof(sign));
        }

        /// <summary>
        /// Validates an array's length.
        /// </summary>
        /// <param name="array">The array to validate</param>
        /// <param name="length">The minimum length requirement</param>
        /// <param name="argument">The name of the array's argument in the factory method</param>
        /// <exception cref="ArgumentException">Validation check failed</exception>
        private static void ValidateArray(FftwArray array, long length, string argument)
        {
            if (array.LongLength < length)
                throw new ArgumentException($"Array length must be at least {length}.", argument);
        }

        /// <summary>
        /// Calculates the expected length of an array used in a complex to complex DFT plan.
        /// <summary/>
        /// <param name="n">The transform size in each dimension</param>
        private static int DftLength(params int[] n)
        {
            int length = 2;
            foreach (int dim in n)
                length *= dim;
            return length;
        }

        /// <summary>
        /// Calculates the expected length of the real array used in a real to complex DFT plan.
        /// <summary/>
        /// <param name="n">The transform size in each dimension</param>
        private static int DftRealLength(params int[] n)
        {
            int length = 1;
            foreach (int dim in n)
                length *= dim;
            return length;
        }

        /// <summary>
        /// Calculates the expected length of the complex array used in a real to complex DFT plan.
        /// <summary/>
        /// <param name="n">The transform size in each dimension</param>
        private static int DftComplexLength(params int[] n)
        {
            int length = 2;
            foreach (int dim in new ArraySegment<int>(n, 0, n.Length - 1))
                length *= dim;
            length *= n[n.Length - 1] / 2 + 1;
            return length;
        }

        /// <summary>
        /// Calculates the expected length of an array used in a real to real transform plan.
        /// <summary/>
        /// <param name="n">The transform size</param>
        /// <param name="kind">The transform kind</param>
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

        /// <summary>
        /// Calculates the expected length of an array used in a real to real transform plan.
        /// <summary/>
        /// <param name="n0">The transform size in dimension 0</param>
        /// <param name="n1">The transform size in dimension 1</param>
        /// <param name="kind0">The transform kind in dimension 0</param>
        /// <param name="kind1">The transform kind in dimension 1</param>
        private static int R2rLength(int n0, int n1, FftwR2rKind kind0, FftwR2rKind kind1) =>
            R2rLength(n0, kind0) * R2rLength(n1, kind1);

        /// <summary>
        /// Calculates the expected length of an array used in a real to real transform plan.
        /// <summary/>
        /// <param name="n0">The transform size in dimension 0</param>
        /// <param name="n1">The transform size in dimension 1</param>
        /// <param name="n2">The transform size in dimension 2</param>
        /// <param name="kind0">The transform kind in dimension 0</param>
        /// <param name="kind1">The transform kind in dimension 1</param>
        /// <param name="kind2">The transform kind in dimension 2</param>
        private static int R2rLength(int n0, int n1, int n2, FftwR2rKind kind0, FftwR2rKind kind1, FftwR2rKind kind2) =>
            R2rLength(n0, kind0) * R2rLength(n1, kind1) * R2rLength(n2, kind2);

        /// <summary>
        /// Calculates the expected length of an array used in a real to real transform plan.
        /// </summary>
        /// <param name="n">The transform size in each dimension</param>
        /// <param name="kind">The transform kind in each dimension</param>
        private static int R2rLength(int[] n, FftwR2rKind[] kind)
        {
            if (n.Length != kind.Length)
                throw new ArgumentException("Length of int[] n and FftwR2rKind[] kind must match.", nameof(kind));
            int length = 1;
            for (int i = 0; i < n.Length; i++)
                length *= R2rLength(n[i], kind[i]);
            return length;
        }

        /// <summary>
        /// Determines whether a transform is to be in-place by comparing array pointers for equality.
        /// </summary>
        private static bool IsInPlace(FftwArray inArray, FftwArray outArray) =>
            inArray.Pointer == outArray.Pointer;
        
        /// <summary>
        /// Gets the method name to use a given new-array execute function
        /// </summary>
        /// <param name="newArrayExecFn">The new-array execute function</param>
        private static string NameForNewArrayExecFn(NewArrayExecFn newArrayExecFn)
        {
            switch (newArrayExecFn)
            {
                case NewArrayExecFn.Dft:
                    return nameof(ExecuteDft);
                case NewArrayExecFn.SplitDft:
                    return nameof(ExecuteSplitDft);
                case NewArrayExecFn.DftR2c:
                    return nameof(ExecuteDftR2c);
                case NewArrayExecFn.SplitDftR2c:
                    return nameof(ExecuteSplitDftR2c);
                case NewArrayExecFn.DftC2r:
                    return nameof(ExecuteDftC2r);
                case NewArrayExecFn.SplitDftC2r:
                    return nameof(ExecuteSplitDftC2r);
                case NewArrayExecFn.R2r:
                    return nameof(ExecuteR2r);
                default:
                    throw new ArgumentOutOfRangeException(nameof(newArrayExecFn));
            }
        }

        /// <summary>
        /// Validates the use of a given new-array execute function
        /// </summary>
        /// <param name="newArrayExecFn">The new-array execute function</param>
        private void ValidateNewArrayExecFn(NewArrayExecFn expected)
        {
            if (constraints.newArrayExecFn != expected)
                throw new InvalidOperationException("Cannot use this new-array execute method for this plan. " +
                    $"Please use {NameForNewArrayExecFn(constraints.newArrayExecFn)} instead.");
        }

        /// <summary>
        /// Checks that arguments are in-place as expected
        /// </summary>
        /// <param name="array0">The first array to check</param>
        /// <param name="array1">The second array to check</param>
        /// <param name="name0">The parameter name for the first array</param>
        /// <param name="name1">The parameter name for the second array</param>
        private void ValidateIsInPlace(FftwArray array0, FftwArray array1, string name0, string name1)
        {
            if (!IsInPlace(array0, array1))
                throw new InvalidOperationException("This plan was created as an in-place transform. " +
                    $"Please pass the same array for both {name0} and {name1}");
        }

        /// <summary>
        /// Validates memory alignment of arrays
        /// </summary>
        /// <param name="arrays">Arrays to test</param>
        private void ValidateAlignment(params FftwArray[] arrays)
        {
            if (mustAlign && arrays.Any(x => !x.FftwAllocated))
                throw new InvalidOperationException("This plan was created without the FFTW_UNALIGNED flag. " +
                    "All array arguments should be allocated by FFTW to ensure proper memory alignment.");
        }

        #endregion

        #region Basic Interface

        #region DFT

        /// <summary>
        /// Creates a 1D DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft_1d.
        /// </summary>
        /// <remarks>
        /// This overload accepts user-supplied arrays which are owned by the user and must be disposed by the user
        /// after the user disposes the plan. Use this overload in less common use cases where the arrays must be user
        /// supplied or their lifetimes user controlled.
        /// </remarks>
        /// <param name="n0">The size of the transform in the 0th dimension</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
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
                new NewArrayExecConstraints(NewArrayExecFn.Dft, IsInPlace(inArray, outArray), length, length),
                inArray, outArray);
        }

        /// <summary>
        /// Creates a 1D DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft_1d.
        /// </summary>
        /// <remarks>
        /// This overload creates arrays which are owned by the plan and are disposed by the plan when the user
        /// disposes the plan. Use this overload for common out of place transform use cases.
        /// </remarks>
        /// <param name="n0">The size of the transform in the 0th dimension</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
        public static FftwPlan Dft1D(int n0, out FftwArray inArray, out FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n0);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return Dft1D(n0, inArray, outArray, sign, flags);
        }

        /// <summary>
        /// Creates a 1D DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft_1d.
        /// </summary>
        /// <remarks>
        /// This overload creates an array which is owned by the plan and is disposed by the plan when the user
        /// disposes the plan. Use this overload for common in place transform use cases.
        /// </remarks>
        /// <param name="n0">The size of the transform in the 0th dimension</param>
        /// <param name="ioArray">The complex data input / output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
        public static FftwPlan Dft1D(int n0, out FftwArray ioArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftLength(n0));
            return Dft1D(n0, ioArray, ioArray, sign, flags);
        }

        /// <summary>
        /// Creates a 2D DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft_2d.
        /// </summary>
        /// <remarks>
        /// This overload accepts user-supplied arrays which are owned by the user and must be disposed by the user
        /// after the user disposes the plan. Use this overload in less common use cases where the arrays must be user
        /// supplied or their lifetimes user controlled.
        /// </remarks>
        /// <param name="n0">The size of the transform in the 0th dimension</param>
        /// <param name="n1">The size of the transform in the 1st dimension</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
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
                new NewArrayExecConstraints(NewArrayExecFn.Dft, IsInPlace(inArray, outArray), length, length),
                inArray, outArray);
        }

        /// <summary>
        /// Creates a 2D DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft_2d.
        /// </summary>
        /// <remarks>
        /// This overload creates arrays which are owned by the plan and are disposed by the plan when the user
        /// disposes the plan. Use this overload for common out of place transform use cases.
        /// </remarks>
        /// <param name="n0">The size of the transform in the 0th dimension</param>
        /// <param name="n1">The size of the transform in the 1st dimension</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
        public static FftwPlan Dft2D(int n0, int n1, out FftwArray inArray, out FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n0, n1);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return Dft2D(n0, n1, inArray, outArray, sign, flags);
        }

        /// <summary>
        /// Creates a 2D DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft_2d.
        /// </summary>
        /// <remarks>
        /// This overload creates an array which is owned by the plan and is disposed by the plan when the user
        /// disposes the plan. Use this overload for common in place transform use cases.
        /// </remarks>
        /// <param name="n0">The size of the transform in the 0th dimension</param>
        /// <param name="n1">The size of the transform in the 1st dimension</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
        public static FftwPlan Dft2D(int n0, int n1, out FftwArray ioArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftLength(n0, n1));
            return Dft2D(n0, n1, ioArray, ioArray, sign, flags);
        }

        /// <summary>
        /// Creates a 3D DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft_3d.
        /// </summary>
        /// <remarks>
        /// This overload accepts user-supplied arrays which are owned by the user and must be disposed by the user
        /// after the user disposes the plan. Use this overload in less common use cases where the arrays must be user
        /// supplied or their lifetimes user controlled.
        /// </remarks>
        /// <param name="n0">The size of the transform in the 0th dimension</param>
        /// <param name="n1">The size of the transform in the 1st dimension</param>
        /// <param name="n2">The size of the transform in the 2nd dimension</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
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
                new NewArrayExecConstraints(NewArrayExecFn.Dft, IsInPlace(inArray, outArray), length, length),
                inArray, outArray);
        }

        /// <summary>
        /// Creates a 3D DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft_3d.
        /// </summary>
        /// <remarks>
        /// This overload creates arrays which are owned by the plan and are disposed by the plan when the user
        /// disposes the plan. Use this overload for common out of place transform use cases.
        /// </remarks>
        /// <param name="n0">The size of the transform in the 0th dimension</param>
        /// <param name="n1">The size of the transform in the 1st dimension</param>
        /// <param name="n2">The size of the transform in the 2nd dimension</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
        public static FftwPlan Dft3D(int n0, int n1, int n2, out FftwArray inArray, out FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n0, n1, n2);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return Dft3D(n0, n1, n2, inArray, outArray, sign, flags);
        }

        /// <summary>
        /// Creates a 3D DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft_3d.
        /// </summary>
        /// <remarks>
        /// This overload creates an array which is owned by the plan and is disposed by the plan when the user
        /// disposes the plan. Use this overload for common in place transform use cases.
        /// </remarks>
        /// <param name="n0">The size of the transform in the 0th dimension</param>
        /// <param name="n1">The size of the transform in the 1st dimension</param>
        /// <param name="n2">The size of the transform in the 2nd dimension</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
        public static FftwPlan Dft3D(int n0, int n1, int n2, out FftwArray ioArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            ioArray = new FftwArray(DftLength(n0, n1, n2));
            return Dft3D(n0, n1, n2, ioArray, ioArray, sign, flags);
        }

        /// <summary>
        /// Creates a DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft.
        /// </summary>
        /// <remarks>
        /// This overload accepts user-supplied arrays which are owned by the user and must be disposed by the user
        /// after the user disposes the plan. Use this overload in less common use cases where the arrays must be user
        /// supplied or their lifetimes user controlled.
        /// </remarks>
        /// <param name="n">The size of the transform in each dimension (the length of n is the rank)</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
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
                new NewArrayExecConstraints(NewArrayExecFn.Dft, IsInPlace(inArray, outArray), length, length),
                inArray, outArray);
        }

        /// <summary>
        /// Creates a DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft.
        /// </summary>
        /// <remarks>
        /// This overload creates arrays which are owned by the plan and are disposed by the plan when the user
        /// disposes the plan. Use this overload for common out of place transform use cases.
        /// </remarks>
        /// <param name="n">The size of the transform in each dimension (the length of n is the rank)</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
        public static FftwPlan Dft(int[] n, out FftwArray inArray, out FftwArray outArray,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            int length = DftLength(n);
            inArray = new FftwArray(length);
            outArray = new FftwArray(length);
            return Dft(n, inArray, outArray, sign, flags);
        }

        /// <summary>
        /// Creates a DFT plan via the basic interface. See FFTW documentation for details on fftw_plan_dft.
        /// </summary>
        /// <remarks>
        /// This overload creates an array which is owned by the plan and is disposed by the plan when the user
        /// disposes the plan. Use this overload for common in place transform use cases.
        /// </remarks>
        /// <param name="n">The size of the transform in each dimension (the length of n is the rank)</param>
        /// <param name="inArray">The complex data input array</param>
        /// <param name="outArray">The complex data output array</param>
        /// <param name="sign">The direction of the transform</param>
        /// <param name="flags">One or more planner flags, combined by bitwise OR</param>
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
            bool isInPlace = IsInPlace(inArray, outArray);
            int inLength = isInPlace ? outLength : DftRealLength(n0);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_r2c_1d(n0,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                new NewArrayExecConstraints(NewArrayExecFn.DftR2c, isInPlace, inLength, outLength),
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
            bool isInPlace = IsInPlace(inArray, outArray);
            int inLength = isInPlace ? outLength : DftRealLength(n0, n1);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_r2c_2d(n0, n1,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                new NewArrayExecConstraints(NewArrayExecFn.DftR2c, isInPlace, inLength, outLength),
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
            bool isInPlace = IsInPlace(inArray, outArray);
            int inLength = isInPlace ? outLength : DftRealLength(n0, n1, n2);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_r2c_3d(n0, n1, n2,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                new NewArrayExecConstraints(NewArrayExecFn.DftR2c, isInPlace, inLength, outLength),
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
            bool isInPlace = IsInPlace(inArray, outArray);
            int inLength = isInPlace ? outLength : DftRealLength(n);
            ValidateArray(inArray, outLength, nameof(inArray));
            ValidateArray(outArray, inLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_r2c(n.Length, n,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                new NewArrayExecConstraints(NewArrayExecFn.DftR2c, isInPlace, inLength, outLength),
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
            bool isInPlace = IsInPlace(inArray, outArray);
            int outLength = isInPlace ? inLength : DftRealLength(n0);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_c2r_1d(n0,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                new NewArrayExecConstraints(NewArrayExecFn.SplitDftC2r, isInPlace, inLength, outLength),
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
            bool isInPlace = IsInPlace(inArray, outArray);
            int outLength = isInPlace ? inLength : DftRealLength(n0, n1);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_c2r_2d(n0, n1,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                new NewArrayExecConstraints(NewArrayExecFn.DftC2r, isInPlace, inLength, outLength),
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
            bool isInPlace = IsInPlace(inArray, outArray);
            int outLength = isInPlace ? inLength : DftRealLength(n0, n1, n2);
            ValidateArray(inArray, inLength, nameof(inArray));
            ValidateArray(outArray, outLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_c2r_3d(n0, n1, n2,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                new NewArrayExecConstraints(NewArrayExecFn.DftC2r, isInPlace, inLength, outLength),
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
            bool isInPlace = IsInPlace(inArray, outArray);
            int outLength = isInPlace ? inLength : DftRealLength(n);
            ValidateArray(inArray, outLength, nameof(inArray));
            ValidateArray(outArray, inLength, nameof(outArray));
            return new FftwPlan(flags,
                fftw_plan_dft_c2r(n.Length, n,
                    inArray.Pointer, outArray.Pointer,
                    (uint_t)flags),
                new NewArrayExecConstraints(NewArrayExecFn.DftC2r, isInPlace, inLength, outLength),
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
                new NewArrayExecConstraints(NewArrayExecFn.R2r, IsInPlace(inArray, outArray), length, length),
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
                new NewArrayExecConstraints(NewArrayExecFn.R2r, IsInPlace(inArray, outArray), length, length),
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
                new NewArrayExecConstraints(NewArrayExecFn.R2r, IsInPlace(inArray, outArray), length, length),
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
                new NewArrayExecConstraints(NewArrayExecFn.R2r, IsInPlace(inArray, outArray), length, length),
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

        #region Advanced Interface

        /*public static FftwPlan ManyDft(int[] n, int howmany,
            FftwArray inArray, int[] inembed, int istride, int idist,
            FftwArray outArray, int[] onembed, int ostride, int odist,
            FftwSign sign = FftwSign.FFTW_FORWARD, FftwFlags flags = FftwFlags.FFTW_MEASURE)
        {
            if (inembed == null)
                inembed = n;
            else
            {

            }
            if (onembed == null)
                onembed = n;
        }*/

        #endregion

        #region Execute

        /// <summary>
        /// Executes the plan on the arrays it was created with. See FFTW documentation for details on fftw_execute.
        /// </summary>
        public void Execute()
        {
            fftw_execute(pointer);
        }

        /// <summary>
        /// Executes the plan on new arrays supplied. See FFTW documentation for details on fftw_execute_dft.
        /// </summary>
        public void ExecuteDft(FftwArray inArray, FftwArray outArray)
        {
            ValidateNewArrayExecFn(NewArrayExecFn.Dft);
            ValidateArray(inArray, constraints.length0, nameof(inArray));
            ValidateArray(outArray, constraints.length1, nameof(outArray));
            if (constraints.inPlace)
                ValidateIsInPlace(inArray, outArray, nameof(inArray), nameof(outArray));
            ValidateAlignment(inArray, outArray);
            fftw_execute_dft(pointer, inArray.Pointer, outArray.Pointer);
        }

        public void ExecuteSplitDft(FftwArray riArray, FftwArray iiArray, FftwArray roArray, FftwArray ioArray)
        {
            ValidateNewArrayExecFn(NewArrayExecFn.Dft);
            ValidateArray(riArray, constraints.length0, nameof(riArray));
            ValidateArray(iiArray, constraints.length1, nameof(iiArray));
            ValidateArray(roArray, constraints.length2, nameof(roArray));
            ValidateArray(ioArray, constraints.length3, nameof(ioArray));
            if (constraints.inPlace)
            {
                ValidateIsInPlace(riArray, roArray, nameof(riArray), nameof(roArray));
                ValidateIsInPlace(iiArray, ioArray, nameof(iiArray), nameof(ioArray));
            }
            ValidateAlignment(riArray, iiArray, roArray, ioArray);
            fftw_execute_split_dft(pointer, riArray.Pointer, iiArray.Pointer, roArray.Pointer, ioArray.Pointer);
        }

        public void ExecuteDftR2c(FftwArray inArray, FftwArray outArray)
        {
            ValidateNewArrayExecFn(NewArrayExecFn.DftR2c);
            ValidateArray(inArray, constraints.length0, nameof(inArray));
            ValidateArray(outArray, constraints.length1, nameof(outArray));
            if (constraints.inPlace)
                ValidateIsInPlace(inArray, outArray, nameof(inArray), nameof(outArray));
            ValidateAlignment(inArray, outArray);
            fftw_execute_dft_r2c(pointer, inArray.Pointer, outArray.Pointer);
        }

        public void ExecuteSplitDftR2c(FftwArray inArray, FftwArray roArray, FftwArray ioArray)
        {
            ValidateNewArrayExecFn(NewArrayExecFn.SplitDftR2c);
            ValidateArray(inArray, constraints.length0, nameof(inArray));
            ValidateArray(roArray, constraints.length1, nameof(roArray));
            ValidateArray(ioArray, constraints.length2, nameof(ioArray));
            if (constraints.inPlace)
                ValidateIsInPlace(inArray, roArray, nameof(inArray), nameof(roArray));
            ValidateAlignment(inArray, roArray, ioArray);
            fftw_execute_split_dft_r2c(pointer, inArray.Pointer, roArray.Pointer, ioArray.Pointer);
        }

        public void ExecuteDftC2r(FftwArray inArray, FftwArray outArray)
        {
            ValidateNewArrayExecFn(NewArrayExecFn.DftC2r);
            ValidateArray(inArray, constraints.length0, nameof(inArray));
            ValidateArray(outArray, constraints.length1, nameof(outArray));
            if (constraints.inPlace)
                ValidateIsInPlace(inArray, outArray, nameof(inArray), nameof(outArray));
            ValidateAlignment(inArray, outArray);
            fftw_execute_dft_c2r(pointer, inArray.Pointer, outArray.Pointer);
        }

        public void ExecuteSplitDftC2r(FftwArray riArray, FftwArray iiArray, FftwArray outArray)
        {
            ValidateNewArrayExecFn(NewArrayExecFn.SplitDftC2r);
            ValidateArray(riArray, constraints.length0, nameof(riArray));
            ValidateArray(iiArray, constraints.length1, nameof(iiArray));
            ValidateArray(outArray, constraints.length2, nameof(outArray));
            if (constraints.inPlace)
                ValidateIsInPlace(riArray, outArray, nameof(riArray), nameof(outArray));
            ValidateAlignment(riArray, iiArray, outArray);
            fftw_execute_split_dft_c2r(pointer, riArray.Pointer, iiArray.Pointer, outArray.Pointer);
        }

        public void ExecuteR2r(FftwArray inArray, FftwArray outArray)
        {
            ValidateNewArrayExecFn(NewArrayExecFn.R2r);
            ValidateArray(inArray, constraints.length0, nameof(inArray));
            ValidateArray(outArray, constraints.length1, nameof(outArray));
            if (constraints.inPlace)
                ValidateIsInPlace(inArray, outArray, nameof(inArray), nameof(outArray));
            ValidateAlignment(inArray, outArray);
            fftw_execute_r2r(pointer, inArray.Pointer, outArray.Pointer);
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Implementation of Dispose pattern.
        /// </summary>
        /// <param name="disposing">True if disposing, false if finalizing.</param>
        private protected virtual void Dispose(bool disposing)
        {
            // protect from multiple disposal
            if (disposed) return;
            if (disposing)
            {
                foreach (var array in arrays)
                {
                    array.DecrementPlans();
                    if (owned)
                        array.Dispose();
                }
                arrays = null;
            }
            fftw_destroy_plan(pointer);
            GC.SuppressFinalize(this);
            disposed = true;
        }

        /// </inheritdoc>
        public void Dispose() => Dispose(true);

        ~FftwPlan()
        {
            Dispose(false);
        }

        #endregion
    }
}
