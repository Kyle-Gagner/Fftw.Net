using System;

namespace Fftw.Net
{

    public abstract class FftwView
    {
        public FftwArray Array { get; }

        private protected FftwView(FftwArray array)
        {
            Array = array;
        }
    }

    public class FftwRealView : FftwView
    {
        internal FftwRealView(FftwArray array) : base(array) { }
        
        /// <summary>
        /// Gets a <see cref="Span<T>"/> for the underlying memory backing the buffer.
        /// </summary>
        /// <remarks>
        /// Due to its safety and efficiency, <see cref="Span<T>"/> is the preferred means of accessing the memory.
        /// However, this property is not available in .NET Framework, only .NET Core. Spans are limited by the size of
        /// a 32-bit integer. To access the span on very large arrays, use <see cref="Slice(long,long)"/> to create a
        /// smaller array backed by the same memory, then use the span on that smaller array.
        /// </remarks>
        /// <exception cref="OverflowException">The length exceeds <see cref="Int32.MaxValue"/>.</exception>
        public Span<double> Span
        {
            get
            {
                unsafe
                {
                    try
                    {
                        return new Span<double>(Array.Pointer.ToPointer(), Array.Length);
                    }
                    catch (OverflowException ex)
                    {
                        throw new OverflowException("The length of the array exceeds Int32.MaxValue." +
                            "Try slicing the array to a reasonable size before getting the Span on the slice.", ex);
                    }
                }
            }
        }
    }
}
