using System;
using System.Runtime.InteropServices;

using fftw_plan = System.IntPtr;
using size_t = System.UIntPtr;
using void_ptr = System.IntPtr;
using double_ptr = System.IntPtr;
using ptrdiff_t = System.Int64;
using int_t = System.Int32;
using uint_t = System.UInt32;
using unsigned_t = System.Int32;
using enum_default_t = System.Int32;

namespace Fftw.Net
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct fftw_iodim
    {
        int_t n;
        int_t i_s;
        int_t o_s;
    }
    
    [StructLayout(LayoutKind.Sequential)]
    internal struct fftw_iodim64
    {
        ptrdiff_t n;
        ptrdiff_t i_s;
        ptrdiff_t o_s;
    }

    public enum FftwSign : int_t
    {
        FFTW_FORWARD  = -1,
        FFTW_BACKWARD = +1
    }

    public enum FftwFlags : uint_t
    {
        FFTW_MEASURE        = 0x000000,
        FFTW_DESTROY_INPUT  = 0x000001,
        FFTW_UNALIGNED      = 0x000002,
        FFTW_EXHAUSTIVE     = 0x000008,
        FFTW_PRESERVE_INPUT = 0x000010,
        FFTW_PATIENT        = 0x000020,
        FFTW_ESTIMATE       = 0x000040,
        FFTW_WISDOM_ONLY    = 0x200000
    }

    public enum FftwR2rKind: enum_default_t
    {
        FFTW_R2HC    = 0,
        FFTW_HC2R    = 1,
        FFTW_DHT     = 2,
        FFTW_REDFT00 = 3,
        FFTW_REDFT01 = 4,
        FFTW_REDFT10 = 5,
        FFTW_REDFT11 = 6,
        FFTW_RODFT00 = 7,
        FFTW_RODFT01 = 8,
        FFTW_RODFT10 = 9,
        FFTW_RODFT11 = 10
    }
    
    internal static class FftwBindings
    {
        [DllImport("libfftw3", EntryPoint = "fftw_malloc")]
        public static extern void_ptr fftw_malloc(size_t n);

        [DllImport("libfftw3", EntryPoint = "fftw_free")]
        public static extern fftw_plan fftw_free(void_ptr ptr);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_1d")]
        public static extern fftw_plan fftw_plan_dft_1d(
            int_t n0,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_2d")]
        public static extern fftw_plan fftw_plan_dft_2d(
            int_t n0, int_t n1,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_3d")]
        public static extern fftw_plan fftw_plan_dft_3d(
            int_t n0, int_t n1, int_t n2,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft")]
        public static extern fftw_plan fftw_plan_dft(
            int_t rank, int_t[] n,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_r2c_1d")]
        public static extern fftw_plan fftw_plan_dft_r2c_1d(
            int_t n0,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_r2c_2d")]
        public static extern fftw_plan fftw_plan_dft_r2c_2d(
            int_t n0, int_t n1,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_r2c_3d")]
        public static extern fftw_plan fftw_plan_dft_r2c_3d(
            int_t n0, int_t n1, int_t n2,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_r2c")]
        public static extern fftw_plan fftw_plan_dft_r2c(
            int_t rank, int_t[] n,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_c2r_1d")]
        public static extern fftw_plan fftw_plan_dft_c2r_1d(
            int_t n0,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_c2r_2d")]
        public static extern fftw_plan fftw_plan_dft_c2r_2d(
            int_t n0, int_t n1,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_c2r_3d")]
        public static extern fftw_plan fftw_plan_dft_c2r_3d(
            int_t n0, int_t n1, int_t n2,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_dft_c2r")]
        public static extern fftw_plan fftw_plan_dft_c2r(
            int_t rank, int_t[] n,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_r2r_1d")]
        public static extern fftw_plan fftw_plan_r2r_1d(
            int_t n0,
            double_ptr in_ptr, double_ptr out_ptr,
            FftwR2rKind kind0,
            uint flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_r2r_2d")]
        public static extern fftw_plan fftw_plan_r2r_2d(
            int_t n0, int_t n1,
            double_ptr in_ptr, double_ptr out_ptr,
            FftwR2rKind kind0, FftwR2rKind kind1,
            uint flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_r2r_3d")]
        public static extern fftw_plan fftw_plan_r2r_3d(
            int_t n0, int_t n1, int_t n2,
            double_ptr in_ptr, double_ptr out_ptr,
            FftwR2rKind kind0, FftwR2rKind kind1, FftwR2rKind kind2,
            uint flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_r2r")]
        public static extern fftw_plan fftw_plan_r2r(int rank, int_t[] n,
            double_ptr in_ptr, double_ptr out_ptr,
            FftwR2rKind[] kind,
            uint flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_many_dft")]
        public static extern fftw_plan fftw_plan_many_dft(
            int_t rank, int_t[] n, int_t howmany,
            double_ptr in_ptr, int_t[] inembed,
            int_t istride, int_t idist,
            double_ptr out_ptr, int_t[] onembed,
            int_t ostride, int_t odist,
            int_t sign, uint flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_many_dft_r2c")]
        public static extern fftw_plan fftw_plan_many_dft_r2c(
            int_t rank, int_t[] n, int_t howmany,
            double_ptr in_ptr, int_t[] inembed,
            int_t istride, int_t idist,
            double_ptr out_ptr, int_t[] onembed,
            int_t ostride, int_t odist,
            int_t sign, uint flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_many_dft_c2r")]
        public static extern fftw_plan fftw_plan_many_dft_c2r(
            int_t rank, int_t[] n, int_t howmany,
            double_ptr in_ptr, int_t[] inembed,
            int_t istride, int_t idist,
            double_ptr out_ptr, int_t[] onembed,
            int_t ostride, int_t odist,
            int_t sign, uint flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_many_r2r")]
        public static extern fftw_plan fftw_plan_many_dft_r2r(
            int_t rank, int_t[] n, int_t howmany,
            double_ptr in_ptr, int_t[] inembed,
            int_t istride, int_t idist,
            double_ptr out_ptr, int_t[] onembed,
            int_t ostride, int_t odist,
            FftwR2rKind[] kind, uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru_dft")]
        public static extern fftw_plan fftw_plan_guru_dft(
            int_t rank, fftw_iodim[] dims,
            int_t howmany_rank, fftw_iodim[] howmany_dims,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru_split_dft")]
        public static extern fftw_plan fftw_plan_guru_split_dft(
            int_t rank, fftw_iodim[] dims,
            int_t howmany_rank, fftw_iodim[] howmany_dims,
            double_ptr ri_ptr, double_ptr ii_ptr,
            double_ptr ro_ptr, double_ptr io_ptr,
            uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru_dft_r2c")]
        public static extern fftw_plan fftw_plan_guru_dft_r2c(
            int_t rank, fftw_iodim[] dims,
            int_t howmany_rank, fftw_iodim[] howmany_dims,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru_split_dft_r2c")]
        public static extern fftw_plan fftw_plan_guru_split_dft_r2c(
            int_t rank, fftw_iodim[] dims,
            int_t howmany_rank, fftw_iodim[] howmany_dims,
            double_ptr in_ptr, double_ptr ro_ptr, double_ptr io_ptr,
            uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru_dft_c2r")]
        public static extern fftw_plan fftw_plan_guru_dft_c2r(
            int_t rank, fftw_iodim[] dims,
            int_t howmany_rank, fftw_iodim[] howmany_dims,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru_split_dft_c2r")]
        public static extern fftw_plan fftw_plan_guru_split_dft_c2r(
            int_t rank, fftw_iodim[] dims,
            int_t howmany_rank, fftw_iodim[] howmany_dims,
            double_ptr in_ptr, double_ptr ii_ptr, double_ptr out_ptr,
            uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru_r2r")]
        public static extern fftw_plan fftw_plan_guru_split_dft_c2r(
            int_t rank, fftw_iodim[] dims,
            int_t howmany_rank, fftw_iodim[] howmany_dims,
            double_ptr in_ptr, double_ptr out_ptr,
            FftwR2rKind[] kind, uint_t flags);

        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru64_dft")]
        public static extern fftw_plan fftw_plan_guru64_dft(
            int_t rank, fftw_iodim64[] dims,
            int_t howmany_rank, fftw_iodim64[] howmany_dims,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru64_split_dft")]
        public static extern fftw_plan fftw_plan_guru64_split_dft(
            int_t rank, fftw_iodim64[] dims,
            int_t howmany_rank, fftw_iodim64[] howmany_dims,
            double_ptr ri_ptr, double_ptr ii_ptr,
            double_ptr ro_ptr, double_ptr io_ptr,
            uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru64_dft_r2c")]
        public static extern fftw_plan fftw_plan_guru64_dft_r2c(
            int_t rank, fftw_iodim64[] dims,
            int_t howmany_rank, fftw_iodim64[] howmany_dims,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru64_split_dft_r2c")]
        public static extern fftw_plan fftw_plan_guru_split_dft_r2c(
            int_t rank, fftw_iodim64[] dims,
            int_t howmany_rank, fftw_iodim64[] howmany_dims,
            double_ptr in_ptr, double_ptr ro_ptr, double_ptr io_ptr,
            uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru64_dft_c2r")]
        public static extern fftw_plan fftw_plan_guru_dft_c2r(
            int_t rank, fftw_iodim64[] dims,
            int_t howmany_rank, fftw_iodim64[] howmany_dims,
            double_ptr in_ptr, double_ptr out_ptr,
            int_t sign, uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru64_split_dft_c2r")]
        public static extern fftw_plan fftw_plan_guru_split_dft_c2r(
            int_t rank, fftw_iodim64[] dims,
            int_t howmany_rank, fftw_iodim64[] howmany_dims,
            double_ptr in_ptr, double_ptr ii_ptr, double_ptr out_ptr,
            uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_plan_guru64_r2r")]
        public static extern fftw_plan fftw_plan_guru64_split_dft_c2r(
            int_t rank, fftw_iodim64[] dims,
            int_t howmany_rank, fftw_iodim64[] howmany_dims,
            double_ptr in_ptr, double_ptr out_ptr,
            FftwR2rKind[] kind, uint_t flags);
        
        [DllImport("libfftw3", EntryPoint = "fftw_execute")]
        public static extern void fftw_execute(fftw_plan ptr);

        [DllImport("libfftw3", EntryPoint = "fftw_destroy_plan")]
        public static extern void fftw_destroy_plan(fftw_plan ptr);

        [DllImport("libfftw3", EntryPoint = "fftw_execute_dft")]
        public static extern void fftw_execute_dft(
            IntPtr p,
            double_ptr in_ptr, double_ptr out_ptr);
        
        [DllImport("libfftw3", EntryPoint = "fftw_execute_split_dft")]
        public static extern void fftw_execute_split_dft(
            IntPtr p,
            double_ptr ri_ptr, double_ptr ii_ptr, double_ptr ro_ptr, double_ptr io_ptr);
        
        [DllImport("libfftw3", EntryPoint = "fftw_execute_dft_r2c")]
        public static extern void fftw_execute_dft_r2c(
            IntPtr p,
            double_ptr in_ptr, double_ptr out_ptr);
        
        [DllImport("libfftw3", EntryPoint = "fftw_execute_split_dft_r2c")]
        public static extern void fftw_execute_split_dft_r2c(
            IntPtr p,
            double_ptr in_ptr, double_ptr ro_ptr, double_ptr io_ptr);
        
        [DllImport("libfftw3", EntryPoint = "fftw_execute_dft_c2r")]
        public static extern void fftw_execute_dft_c2r(
            IntPtr p,
            double_ptr in_ptr, double_ptr out_ptr);
        
        [DllImport("libfftw3", EntryPoint = "fftw_execute_split_dft_c2r")]
        public static extern void fftw_execute_split_dft_c2r(
            IntPtr p,
            double_ptr ri_ptr, double_ptr ii_ptr, double_ptr out_ptr);
        
        [DllImport("libfftw3", EntryPoint = "fftw_execute_r2r")]
        public static extern void fftw_execute_r2r(
            IntPtr p,
            double_ptr in_ptr, double_ptr out_ptr);
        
        [DllImport("libfftw3", EntryPoint = "fftw_alignment_of")]
        public static extern int_t fftw_alignment_of(double_ptr array_ptr);
    }
}
