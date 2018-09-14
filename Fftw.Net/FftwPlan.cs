using System;
using static Fftw.Net.FftwBindings;

namespace Fftw.Net
{
    public abstract class FftwPlan : IDisposable
    {
        protected const string align_msg = "This plan was not constructed with the FFTW_UNALIGNED flag, all new array executes must be on aligned arrays";

        private bool disposed = false;

        private IntPtr plan_ptr;

        protected IntPtr PlanPtr { get { return plan_ptr; } private set { plan_ptr = value; } }

        protected void Initialize(IntPtr plan_ptr)
        {
            PlanPtr = plan_ptr;
        }

        public void Execute()
        {
            fftw_execute(PlanPtr);
        }

        private void Dispose(bool disposing)
        {
            fftw_destroy_plan(PlanPtr);
        }

        public void Dispose()
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

        ~FftwPlan()
        {
            if (!disposed)
            {
                disposed = true;
                Dispose(false);
            }
        }
    }
}
