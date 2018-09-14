using System;
using static Fftw.Net.FftwBindings;

namespace Fftw.Net
{
    public class FftwPlanDft : FftwPlan
    {
        private FftwArray in_array, out_array;

        private bool must_align;

        public FftwArray InArray { get { return in_array; } private set { in_array = value; }}

        public FftwArray OutArray { get { return out_array; } private set { out_array = value; }}

        public FftwPlanDft(int n0, FftwSign sign, FftwFlags flags, bool inplace)
        {
            if ((int)sign != 1 && (int)sign != -1)
                throw new ArgumentOutOfRangeException("sign");
            if (n0 <= 0)
                throw new ArgumentOutOfRangeException("n0");
            InArray = new FftwArray(n0 * 2);
            if (inplace) OutArray = InArray;
            else out_array = new FftwArray(n0 * 2);
            must_align = (flags & FftwFlags.FFTW_UNALIGNED) == 0;
            unsafe
            {
                Initialize(fftw_plan_dft_1d(
                    n0,
                    (IntPtr)InArray, (IntPtr)OutArray,
                    (int)sign, (uint)flags));
            }
        }

        public void Execute(FftwArray in_array, FftwArray out_array)
        {
            unsafe
            {
                if (must_align)
                    if (fftw_alignment_of((IntPtr)InArray) != fftw_alignment_of((IntPtr)this.InArray) ||
                        fftw_alignment_of((IntPtr)OutArray) != fftw_alignment_of((IntPtr)this.OutArray))
                        throw new ArgumentException(align_msg);
                fftw_execute_dft(PlanPtr, (IntPtr)InArray, (IntPtr)OutArray);
            }
        }
    }
}
