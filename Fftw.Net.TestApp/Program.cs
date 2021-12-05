using System;
using Fftw.Net;

namespace Fftw.Net.TestApp
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("FFTW Test:");
            var foo = FftwPlan.Dft1D(4, out FftwArray ioArray);
            var bar = ioArray.DoubleSpan;
            bar[0] =  1;
            bar[1] =  0;
            bar[2] =  0;
            bar[3] = -1;
            bar[4] = -1;
            bar[5] =  0;
            bar[6] =  0;
            bar[7] =  1;
            foo.Execute();
            PrintArry(ioArray);
        }
        static void PrintArry(FftwArray arry)
        {
            var foo = arry.ComplexSpan;
            foreach (var value in foo)
                Console.WriteLine($"<{value.Real}, {value.Imaginary}>");
        }
    }
}
